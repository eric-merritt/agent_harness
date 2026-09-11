# Mixed-Precision Quantization Pipeline — Plan

## Context

`prompt.md` specifies a mixed-precision quantization pipeline for an RTX 4080 Super,
implemented **entirely within the `memory_controller` module** (plus one new shader next
to the existing one). The goal is a working, verifiable pipeline that:

1. Manages exactly **two descriptor bindings** across an alternating layer loop — a
   ping-pong activation/scratch pair and a paged FP16 weight-tile space sized for both
   square attention layers and wide SwiGLU FFN expansions (no hardcoded dims).
2. Estimates per-layer Hessian sensitivity on the fly with **Hutchinson's estimator**
   (`Hv = Xᵀ(Xv)`), using on-the-fly Rademacher vectors, 32-thread subgroup shuffles,
   and atomic scratch accumulation — no dense matrices, no second-order AD.

Task 1 (dual-binding host architecture) is fully specified with exact byte counts I can
verify against. Task 2's Trellis bit-allocation loop is deliberately the **last** piece:
this plan builds the full pipeline up to and including a real directional-error-variance
→ per-tile bit-rate map, then stops at a clearly-marked interface so we review
understanding before implementing the full trellis DP.

## Knowledge boundary (per user)

`src/memory_controller/` is the knowledge/search boundary. The only files outside it I
touch are those that define the pipeline contract I'm extending:
- `src/models/sandbag_quantize.comp` — read-only reference for shader conventions
  (push-constant layout, arena-as-flat-binding style).
- `src/models/quantize.rs::dispatch_quantize_shader` — read-only reference for how a
  two-phase dispatch + fence + download is done today.

No broad codebase sweeps.

## Files to create

### 1. `src/memory_controller/pingpong.rs` (NEW) — Task 1 host architecture

A self-contained module that owns the ping-pong allocation and the descriptor layout.
Everything is a **formula** of `(n_tokens, n_dim, n_ffn)` — nothing hardcoded. It reuses
the existing sparse arena (`self.arena`) for backing so it inherits commit/evict for free.

```rust
/// Square layer geometry + wide FFN expansion, all derived from a baseline hidden dim.
#[derive(Clone, Copy, Debug)]
pub struct LayerGeometry { pub n_tokens: u32, pub n_dim: u32, pub n_ffn: u32 }

impl LayerGeometry {
    /// SwiGLU intermediate = (8/3)·n_dim rounded up to a multiple of 32.
    /// 5120 -> 17408. Both dims stay divisible by 32 for the 32×32 tile grid.
    pub fn from_baseline(n_tokens: u32, n_dim: u32) -> Self {
        let n_ffn = ((8 * n_dim) / 3).div_ceil(32) * 32;
        Self { n_tokens, n_dim, n_ffn }
    }
    /// Role 1 activation size (bytes): tokens × n_dim × 4.  5120×4096 -> 83,886,080.
    pub fn activation_bytes(&self) -> u64 { self.n_tokens as u64 * self.n_dim as u64 * 4 }
    /// Square attention weight tile (bytes): n_dim² × 2.  5120²×2 -> 52,428,800.
    pub fn attn_weight_bytes(&self) -> u64 { self.n_dim as u64 * self.n_dim as u64 * 2 }
    /// FFN up/gate weight tile (bytes): n_dim × n_ffn × 2.  5120×17408×2 -> 178,257,920.
    pub fn ffn_weight_bytes(&self) -> u64 { self.n_dim as u64 * self.n_ffn as u64 * 2 }
    /// Tile grid (cols = in/32, rows = out/32). attn: 160×160. ffn_up/gate: 160×544.
    pub fn tile_grid(&self, n_in: u32, n_out: u32) -> (u32, u32) { (n_in/32, n_out/32) }
}

/// Ping-pong activation pair. Two identical pre-allocated blocks; `active` flips per layer.
pub struct PingPongActivation {
    pub buf_a: vk::Buffer, pub buf_b: vk::Buffer,
    pub active: u8,          // 0 -> A is input / B is scratch; 1 -> swapped
    pub size_bytes: u64,
}
impl PingPongActivation {
    /// The buffer currently holding READ-ONLY input activations (Role 1).
    pub fn read_buffer(&self) -> vk::Buffer { if self.active == 0 { self.buf_a } else { self.buf_b } }
    /// The buffer currently acting as WRITABLE SCRATCH accumulator (Role 2), zeroed by fill.
    pub fn scratch_buffer(&self) -> vk::Buffer { if self.active == 0 { self.buf_b } else { self.buf_a } }
    pub fn swap(&mut self) { self.active ^= 1; }
}
```

Plus a `DualBindingLayout` that, given a `LayerGeometry`, returns the two
`(vk::Buffer, vk::DeviceSize)` descriptors (binding 0 = current read/scratch pair,
binding 1 = weight tile region in the sparse arena) and the push-constant values. The
allocation of buf_a/buf_b reuses `GpuContext`'s staging/upload path at construction; the
weight-tile region is carved out of `arena.sparse_buffer` (commit pages first, exactly as
`quantize_gpu` does with `dst_base`).

**Verification hooks:** a `#[cfg(test)]` unit test asserting the three byte counts and both
tile grids against the exact prompt numbers (83,886,080 / 52,428,800 / 178,257,920 and
160×160 / 160×544). These run headless — no GPU needed.

### 2. `src/models/hessian_estimate.comp` (NEW) — Task 2 shader (next to sandbag)

GLSL compute, Vulkan 1.3, following `sandbag_quantize.comp` conventions:
- `layout(local_size_x = 32, local_size_y = 4) in;` — one warp per row of the tile grid.
- Two storage bindings matching Task 1: `binding=0` activation/scratch (f32),
  `binding=1` weight tiles (f16). Push constants carry `n_tokens, n_dim, n_ffn, layer_idx,
  seed`, and the scratch/output offsets so it stays one-immutable-set-friendly.
- **On-the-fly Rademacher vector:** Philox-4x32 seeded from push constants, one bit per
  element — no host upload. `sign = (philox(seed, i) & 1) ? +1.0 : -1.0`.
- **Subgroup shuffle coalescing:** each thread loads ONE activation scalar,
  `subgroupBroadcast`/shuffle to spread it across the warp's registers, then do the
  sign-flip dot against the 32×32 tile held in registers (the "RF" load).
- **Hutchinson product** `Hv = Xᵀ(Xv)`: one pass computes `Xv` into scratch via
  `atomicAdd`, a second dispatch reads it back and accumulates `Xᵀ(Xv)` per tile.
- **Output:** per-tile directional error variance (E[‖h·δw‖²] over the Rademacher draws)
  written to a small f32 plane — this is the input to the bit-rate map.

### 3. `src/memory_controller/hessian.rs` (NEW) — Task 2 pipeline + Trellis interface

Mirrors `create_quantize_pipeline` / `dispatch_quantize_shader` but for the estimator:
- `create_hessian_pipeline(device, act_buf, weight_region)` → build a **two-binding**
  descriptor set layout (STORAGE_BUFFER ×2), push-constant range, pipeline. Same pooled
  cmd/fence discipline as the existing code (`alloc_cmd_buffer`, `alloc_fence`).
- `dispatch_hessian(&ctrl, &geometry, layer_idx, seed)` — fill scratch with
  `vkCmdFillBuffer` (Role 2 zero-init), dispatch pass 1 (Xv→scratch atomics), a
  `VkBufferMemoryBarrier` RAW choke point, dispatch pass 2 (Xᵀ(Xv) → variance plane),
  fence, download the variance plane.
- **Trellis interface (the stopping point):**
  ```rust
  /// Per-tile bit-rate decision from directional error variance.
  /// Lower bits where variance is low (insensitive path), higher bits where the
  /// Rademacher projection sees a sharp gradient. Returns one u8 bit-width per tile.
  pub fn trellis_bit_assignment(variance: &[f32], budget_bits_per_elem: f64) -> Vec<u8> {
      // STUB for now: proportional water-filling placeholder that respects the budget.
      // The full trellis DP (state = remaining budget, transition = per-tile bits,
      // cost = directional error variance × quantization step) lands in a follow-up
      // once this pipeline is verified end-to-end.
  }
  ```
  This is the explicit checkpoint: implement + verify everything above it, then pause here
  for review before the real trellis DP.

## Files to edit (existing, minimal)

- `src/memory_controller/mod.rs` — add `pub mod pingpong; pub mod hessian;`.
- `src/memory_controller/controller.rs` — nothing structural required; reuse
  `GpuContext`, `QuantizePushConstants` pattern, and the sparse arena. If I need a second
  cached pipeline slot on `GpuContext`, add `cached_hessian_pipeline` / `_layout` /
  `_descriptor_set` fields (same shape as the quantize ones) — only if the standalone
  module approach is cleaner, otherwise keep them local to `hessian.rs`.

## Reuse (do NOT reinvent)

- Sparse arena + commit/evict: `virtual_tensor_arena.rs::VirtualTensorArena` (commit_page,
  evict_page). Weight-tile region = committed pages of `arena.sparse_buffer`.
- Upload/download/pooled cmd+fence: `controller.rs::GpuContext::{upload, download,
  batch_upload, alloc_cmd_buffer, alloc_fence, recycle_*}`.
- Push-constant + two-binding pipeline construction pattern: `create_quantize_pipeline`
  (single binding today) and the dispatch/fence/download skeleton in
  `quantize.rs::dispatch_quantize_shader`.

## Verification

1. **Headless unit tests** (`cargo test memory_controller::pingpong`): assert the three
   byte counts and both tile grids match the prompt exactly; assert buf_a/buf_b swap
   alternates roles across N simulated layers.
2. **GPU end-to-end** (requires Vulkan + 4080 Super, same gate as existing quantize):
   - Build both `.spv` with `glslangValidator --target-env vulkan1.3`.
   - Init via existing `init_global_controller`, construct `PingPongActivation` +
     `DualBindingLayout` for the 5120 baseline, run one layer pass of
     `dispatch_hessian`, and confirm the downloaded variance plane is finite/non-zero
     and its length equals the tile count (25,600 for attention).
   - Confirm `trellis_bit_assignment` returns per-tile widths that sum within budget.

## Out of scope / deferred

- The full trellis DP bit-allocation loop — deliberately gated behind review after the
  pipeline is verified (per user: "we'll check your understanding before moving forward").
- Any change to `sandbag_quantize.comp` or the existing sandbag format — untouched.

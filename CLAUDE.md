# CLAUDE.md — agent_harness

## Project Overview

Rust application for ML model weight compression (deduplication) and inference.
Compresses transformer weights using prefix/tail splitting + dedup, with optional
AVX-512 and GPU acceleration. Supports loading GGUF/HF models and running inference
on compressed weights.

---

## Compression Pipeline

**Goal:** Compress f32 model weights into a small deduplicated representation
(prefix/tail split) that can be losslessly or near-losslessly reconstructed.

### How It Works

1. **Quantization** (`src/models/dedupe/truncation.rs`)
   - `quantize_block()` — Fast O(n) percentile-clip. Finds 99.5th percentile of
     absolute values, uses that as clip threshold. Returns `(scale, outliers)`.
   - `quantize_block_avx512()` — Same logic, AVX-512 SIMD for outlier detection
     (16 elements/iteration). Falls back to scalar if AVX-512 unavailable.
   - `quantize_block_kl()` — KL-divergence search. Builds histogram of true
     distribution, tries 10 clip percentiles (0.90–0.999), picks the one with
     minimum KL divergence. Higher quality, more expensive.
   - All three return `(f32, Vec<(usize, f32)>)` — scale and outliers only.
     Quantized i16 values are discarded; caller reconstructs from original f32.

2. **Prefix/Tail Split + Dedup** (`src/models/dedupe/compressor.rs`)
   - `build_from_quantized()` — For each weight:
     ```
     prefix_int = floor(abs(w) * 10^prefix_digits)  // u8, max 256 unique
     tail_int   = round((abs(w) - prefix_int/10^prefix_digits) * 10^7)  // u32
     ```
   - Deduplicates: unique_prefixes (Vec<u8>), unique_tails (Vec<u32>).
   - Builds manifest: Vec<(prefix_idx, tail_idx)> — one entry per element.
   - Builds sign bitvector: 1 bit per element (1=negative, 0=positive).
   - Computes precision loss in the same pass (no second loop).
   - Warns if avg loss > 1% of scale.

3. **Public Entry Points** (`compressor.rs`)
   - `compress_quantized()` — Scalar percentile clip.
   - `compress_quantized_avx512()` — AVX-512 percentile clip.
   - `compress_quantized_kl()` — Scalar KL divergence.
   - `compress_avx512_percent()` / `compress_avx512_kl()` — Convenience wrappers.
   - `compress_from_gpu_percent()` / `compress_from_gpu_kl()` — GPU-reconstructed
     weights → quantize. AVX-512 reconstruction when available.
   - `compress_from_gpu_scalar_percent()` / `compress_from_gpu_scalar_kl()` — Same,
     scalar reconstruction only.
   - `compress_gpu_with_avx512_percent()` / `compress_gpu_with_scalar_tails_percent()` —
     Try GPU first, fall back to scalar.

4. **Chunked Compression** (`compress_job()`)
   - For tensors > CHUNK_SIZE, splits into chunks, compresses each independently,
     concatenates serialized output.
   - Supports `FullPrecision` (raw f32) and `HalfPrecision` (f16) for tensors
     that should not be deduped.

### GPU Path

`crate::gpu::gpu_compute()` returns GPU-extracted `prefix_ints`, `tails`, `signs`.
Compressor reconstructs f32 from GPU output, then runs quantization + dedup.

---

## Decompression Pipeline

**File:** `src/models/dedupe/decompressor.rs`

```
abs_w = (prefix_int as f32) / 10^prefix_digits + (tail_int as f32) / 10^7
w = if sign_bit_set { -abs_w } else { abs_w }
```

Then restores outlier positions at full precision.

- `decompress_all()` — Full reconstruction from Sandbag.
- `decompress_all_global()` — Stub, delegates to `decompress_all()`.
  Global table integration (cross-tensor dedup) not yet implemented.

---

## Sandbag File Format

**Files:** `src/models/dedupe/types.rs`, `src/models/dedupe/serialization.rs`

### Sandbag Struct

| Field | Type | Meaning |
|-------|------|---------|
| `scale` | f32 | Per-block scale factor (reserved) |
| `outliers` | Vec<(usize, f32)> | Outlier positions and original values |
| `count` | usize | Total element count |
| `prefix_digits` | usize | Prefix digits used |
| `unique_prefixes` | Vec<u8> | Deduped unique prefix_int values |
| `unique_tails` | Vec<u32> | Deduped unique tail_int values |
| `manifest` | Vec<(u16, u16)> | Per-element (prefix_idx, tail_idx) |
| `signs` | Vec<u8> | Sign bitvector |

### Binary Layout (little-endian, no padding)

```
┌──────────────────────────────────────────────────────────────┐
│  Header (fixed 8 bytes)                                      │
│  id: 4B ("SBAG" — 0x53424147)                               │
│  version: 2B (major:minor, currently 0x00_03)                │
│  reserved: 2B (padding)                                      │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│  Payload sections (length-prefixed, sequential — NOT offset)  │
│                                                               │
│  [A] count: u32                                               │
│  [B] scale: f32                                               │
│  [C] prefix_digits: u32                                       │
│  [D] outliers: u32(count) × [u32(pos) + f32(val)]            │
│  [E] unique_prefixes: u32(count) × [u8]                      │
│  [F] unique_tails: u32(count) × [u32]                        │
│  [G] manifest: u32(count) × [u16(p_idx) + u16(t_idx)]       │
│  [H] signs: u32(bytes) × [u8] (bitvector, 1 bit per element) │
└──────────────────────────────────────────────────────────────┘
```

Serialization: `sandbag.to_bytes()` / `Sandbag::from_bytes(data)`.

---

## Inference Status

### What Exists

**Model loading** (`src/inference/config.rs`) — `ModelConfig` supports Qwen3.5
(GGUF) and Qwen2 (HuggingFace) formats. Parses from GGUF metadata or config.json.
Handles SSM parameters, RoPE config, layer types, recurrent mask.

**InferenceEngine** (`src/inference/mod.rs`) —
- `open()` / `open_with_progress()` — Loads model, allocates KV caches and SSM
  states per layer, validates tensor names.
- `decompress_all_parallel()` — Multi-threaded decompression, writes temp file
  with 4-bit quantization for non-full-precision tensors.
- `finalize_mmap()` — Memory-maps temp file for zero-copy access.
- `forward()` — Full forward pass through one token.
- `generate()` — Token-by-token generation with temperature sampling.

**Forward pass** implements:
1. RMSNorm (attn_norm)
2. GEMV for Q/K/V via 4-bit quantized weights
3. Per-head Q/K normalization (Qwen3.5's q_norm/k_norm)
4. Multi-section RoPE
5. KV cache storage
6. Multi-head attention with softmax
7. Output projection (attn_output.weight)
8. Residual add
9. RMSNorm (ffn_norm)
10. SwiGLU FFN via 4-bit quantized weights
11. Residual add
12. Final output norm + logits

**Math** (`src/inference/math.rs`) — GEMV (f32 and 4-bit), RMSNorm, L2Norm,
SiLU, sigmoid, softplus, softmax, RoPE (multi-section), depthwise conv1d.
4-bit GEMV has AVX-512 accelerated path + scalar fallback. Parallelizes across
threads for large matrices.

**FFN** (`src/inference/ffn.rs`) — SwiGLU, f32 and 4-bit paths.

**SSM** (`src/inference/ssm.rs`) — Gated Delta Net (linear attention) for
Qwen3.5 recurrent layers. Full implementation with conv1d state, per-head state
matrices, decay, outer product update.

**Sampling** (`src/inference/sampling.rs`) — argmax and temperature + top-k
with xorshift PRNG.

**KV Cache** (`src/inference/kv_cache.rs`) — KV cache allocation.

**Tokenizer** (`src/inference/tokenizer.rs`) — Tokenizer loading.

### What Works

- Complete forward pass for attention layers (Qwen2 style).
- 4-bit quantized weight support (on-the-fly dequantization in GEMV).
- AVX-512 accelerated 4-bit GEMV.
- KV cache.
- Temperature sampling.

### What's Incomplete

- **SSM/recurrent layer path not wired in.** `SsmBlock` struct exists with weight
  references but there's no code that loads SSM weights from the model file. The
  `is_recurrent` mask is computed in the layer loop but the SSM forward path is
  not called. For Qwen3.5 with SSM layers, the recurrent forward pass needs wiring.
- `decompress_all_global()` is a stub — global table integration for cross-tensor
  dedup is not implemented.
- `HalfPrecision` (f16) compression path exists but f16 decompression is not
  implemented in the inference path.

### Assessment

Inference is **substantially implemented** for attention-only models (Qwen2).
For Qwen3.5 with SSM layers, the recurrent forward pass needs to be connected.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models/dedupe/compressor.rs` | Compression entry points, build_from_quantized |
| `src/models/dedupe/decompressor.rs` | Reconstruction from Sandbag |
| `src/models/dedupe/truncation.rs` | quantize_block, quantize_block_avx512, quantize_block_kl |
| `src/models/dedupe/types.rs` | Sandbag, UniqueTail, GlobalTable structs |
| `src/models/dedupe/serialization.rs` | Sandbag binary format (to_bytes/from_bytes) |
| `src/models/dedupe/tensor.rs` | DedupCountTensor struct |
| `src/models/avx512_kernel.rs` | AVX-512 GPU reconstruction helpers |
| `src/inference/mod.rs` | InferenceEngine, forward(), generate() |
| `src/inference/math.rs` | GEMV, RMSNorm, RoPE, etc. |
| `src/inference/ssm.rs` | SSM block (not wired in) |
| `benches/compress_bench.rs` | Compression benchmarks |
| `benches/decompress_bench.rs` | Decompression benchmarks with MSE |

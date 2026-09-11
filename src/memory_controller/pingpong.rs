//! Task 1 — Dual-Binding Sequential Memory Architecture with FFN Expansion Handlers.
//!
//! Owns the host-side allocation + descriptor layout for an alternating layer loop:
//!
//!   * **binding 0** — a ping-pong activation pair (Buffer A / Buffer B). One buffer is
//!     read-only input activations, the other is a writable scratch accumulator; they
//!     swap roles every layer so a layer's clean output is already sitting in the right
//!     slot for the next one.
//!   * **binding 1** — a paged FP16 weight-tile region carved out of the sparse arena,
//!     sized for either a square attention layer or a wide SwiGLU FFN expansion.
//!
//! Nothing here hardcodes a model dimension: every size is a formula of
//! `(n_tokens, n_dim)` (and the derived `n_ffn`). The pure arithmetic lives in
//! [`LayerGeometry`] and is unit-tested headless; the Vulkan allocation wraps it.

use ash::vk;
use std::sync::{Arc, Mutex};

/// Layer geometry: a baseline hidden dimension plus the sequence length, from which
/// every size in this module is derived. `n_ffn` is the SwiGLU intermediate width.
#[derive(Clone, Copy, Debug)]
pub struct LayerGeometry {
	pub n_tokens: u32,
	pub n_dim: u32,
	pub n_ffn: u32,
}

impl LayerGeometry {
	/// Purely model-agnostic constructor. Dims are passed explicitly from the loaded
	/// model's configuration metadata — we never guess an expansion ratio. Both tensor
	/// dims must already be aligned to the 32-element tile boundary or the grid would
	/// not divide cleanly.
	pub fn new(n_tokens: u32, n_dim: u32, n_ffn: u32) -> Self {
		assert_eq!(
			n_dim % 32,
			0,
			"Attention hidden dimension must be aligned to 32-element tile boundaries."
		);
		assert_eq!(
			n_ffn % 32,
			0,
			"FFN intermediate dimension must be aligned to 32-element tile boundaries."
		);

		Self {
			n_tokens,
			n_dim,
			n_ffn,
		}
	}

	/// Role-1 activation size in bytes: `tokens × n_dim × 4` (FP32).
	/// For 5120-wide at 4096 tokens this is exactly 83,886,080.
	pub fn activation_bytes(&self) -> u64 {
		self.n_tokens as u64 * self.n_dim as u64 * 4
	}

	/// Square attention weight-tile size in bytes: `n_dim² × 2` (FP16).
	/// For a 5120×5120 layer this is exactly 52,428,800.
	pub fn attn_weight_bytes(&self) -> u64 {
		self.n_dim as u64 * self.n_dim as u64 * 2
	}

	/// One FFN up/gate weight-tile size in bytes: `n_dim × n_ffn × 2` (FP16).
	/// For the 5120 baseline this is exactly 178,257,920.
	pub fn ffn_weight_bytes(&self) -> u64 {
		self.n_dim as u64 * self.n_ffn as u64 * 2
	}

	/// Tile grid `(cols, rows)` where `cols = n_in/32`, `rows = n_out/32`.
	/// Attention (5120×5120) → (160, 160). FFN up/gate (5120×17408) → (160, 544).
	pub fn tile_grid(&self, n_in: u32, n_out: u32) -> (u32, u32) {
		(n_in / 32, n_out / 32)
	}

	/// Number of 32×32 tiles in the attention layer.
	pub fn attn_tile_count(&self) -> u64 {
		let (c, r) = self.tile_grid(self.n_dim, self.n_dim);
		c as u64 * r as u64
	}

	/// Number of 32×32 tiles in one FFN up/gate tensor.
	pub fn ffn_tile_count(&self) -> u64 {
		let (c, r) = self.tile_grid(self.n_dim, self.n_ffn);
		c as u64 * r as u64
	}
}

/// The ping-pong activation pair: two identical pre-allocated blocks whose roles swap
/// layer by layer. `active == 0` → A is the read input and B is the scratch; after a
/// [`swap`](Self::swap) they exchange.
#[derive(Clone, Copy, Debug)]
pub struct PingPongActivation {
	pub buf_a: vk::Buffer,
	pub buf_b: vk::Buffer,
	/// 0 → A=input / B=scratch; 1 → swapped. Flipped once per layer pass.
	pub active: u8,
	/// Bytes in each block — must equal [`LayerGeometry::activation_bytes`].
	pub size_bytes: u64,
}

impl PingPongActivation {
	/// The buffer currently holding READ-ONLY input activations (Role 1).
	pub fn read_buffer(&self) -> vk::Buffer {
		if self.active == 0 {
			self.buf_a
		} else {
			self.buf_b
		}
	}

	/// The buffer currently acting as the WRITABLE scratch accumulator (Role 2), which
	/// the host zeroes with `vkCmdFillBuffer` at the start of each layer pass.
	pub fn scratch_buffer(&self) -> vk::Buffer {
		if self.active == 0 {
			self.buf_b
		} else {
			self.buf_a
		}
	}

	/// End-of-layer hand-off: after the RAW barrier promotes this layer's scratch into
	/// clean input, flip so the next layer reads from it and accumulates into the other.
	pub fn swap(&mut self) {
		self.active ^= 1;
	}
}

/// The two descriptor bindings for one layer pass, plus the push-constant values the
/// shader needs to address them. Built from a [`LayerGeometry`] so it is correct for
/// both square attention and wide FFN layers with no per-layer branching on the host.
#[derive(Clone, Copy, Debug)]
pub struct DualBindingLayout {
	/// binding 0 — read-only input activations (this layer's ping buffer).
	pub act_buffer: vk::Buffer,
	/// binding 0 — writable scratch accumulator (this layer's pong buffer), zeroed by fill.
	pub scratch_buffer: vk::Buffer,
	/// binding 1 — paged FP16 weight-tile region inside the sparse arena.
	pub weight_region: vk::Buffer,
	/// Byte offset of this layer's weight tiles within `weight_region`.
	pub weight_offset: vk::DeviceSize,
	/// Bytes in the activation block (both ping and pong are this large).
	pub act_bytes: u64,
}

impl DualBindingLayout {
	/// Assemble the two bindings for one layer.
	///
	/// * `act` — the current ping-pong pair; its read/scratch roles are taken as-is.
	/// * `weight_region` + `weight_offset` — where this layer's FP16 tiles live in the
	///   sparse arena (the caller commits those pages first, exactly as `quantize_gpu`
	///   commits its destination pages).
	pub fn new(
		act: &PingPongActivation,
		weight_region: vk::Buffer,
		weight_offset: vk::DeviceSize,
	) -> Self {
		Self {
			act_buffer: act.read_buffer(),
			scratch_buffer: act.scratch_buffer(),
			weight_region,
			weight_offset,
			act_bytes: act.size_bytes,
		}
	}
}

/// Allocate the ping-pong activation pair on the GPU.
///
/// Reuses the arena's allocator (device-local) for two identical blocks of
/// `geometry.activation_bytes()`. Returns the pair ready to be driven by
/// [`DualBindingLayout`]. Kept separate from the pure geometry so the byte-count math
/// stays testable without a Vulkan device.
pub unsafe fn alloc_ping_pong(
	device: &ash::Device,
	allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
	geometry: &LayerGeometry,
) -> Result<PingPongActivation, String> {
	use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};

	let size = geometry.activation_bytes() as vk::DeviceSize;

	unsafe fn make_block(
		device: &ash::Device,
		allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
		size: vk::DeviceSize,
		name: &str,
	) -> Result<vk::Buffer, String> {
		let buf = unsafe {
			device
				.create_buffer(
					&vk::BufferCreateInfo::default()
						.size(size)
						.usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST),
					None,
				)
		}
		.map_err(|e| format!("create ping-pong block failed: {e:?}"))?;

		let mem_reqs = unsafe { device.get_buffer_memory_requirements(buf) };
		let mut guard = allocator.lock().map_err(|e| format!("allocator poisoned: {e}"))?;
		let alloc = guard
			.allocate(&AllocationCreateDesc {
				name,
				requirements: mem_reqs,
				location: gpu_allocator::MemoryLocation::GpuOnly,
				linear: true,
				allocation_scheme: AllocationScheme::GpuAllocatorManaged,
			})
			.map_err(|e| format!("allocate ping-pong block failed: {e}"))?;
		drop(guard);

		unsafe { device
			.bind_buffer_memory(buf, alloc.memory(), alloc.offset()) }
			.map_err(|e| format!("bind ping-pong block failed: {e:?}"))?;

		// The allocation is intentionally leaked into the arena's lifetime; the device
		// owns the memory until it is destroyed. (Matches how the sparse arena keeps
		// its per-page allocations alive in `VirtualPage::gpu_allocation`.)
		let _ = alloc;
		Ok(buf)
	}

	let buf_a = unsafe { make_block(device, allocator, size, "pingpong_A") }?;
	let buf_b = unsafe { make_block(device, allocator, size, "pingpong_B") }?;

	Ok(PingPongActivation {
		buf_a,
		buf_b,
		active: 0,
		size_bytes: geometry.activation_bytes(),
	})
}

#[cfg(test)]
mod tests {
	use super::*;
	use ash::vk::Handle;

	/// The exact numbers from the spec — a regression guard for the sizing formulas.
	#[test]
	fn prompt_byte_counts_and_grids() {
		let g = LayerGeometry::new(4096, 5120, 17408);

		assert_eq!(g.activation_bytes(), 83_886_080, "tokens×dim×4 (FP32 activations)");
		assert_eq!(g.attn_weight_bytes(), 52_428_800, "dim²×2 (FP16 attention tiles)");
		assert_eq!(g.ffn_weight_bytes(), 178_257_920, "dim×ffn×2 (FP16 FFN up/gate)");

		assert_eq!(g.tile_grid(5120, 5120), (160, 160), "attention grid");
		assert_eq!(g.tile_grid(5120, 17408), (160, 544), "FFN up/gate grid");

		assert_eq!(g.attn_tile_count(), 25_600, "160×160 attention tiles");
		assert_eq!(g.ffn_tile_count(), 87_040, "160×544 FFN tiles");
	}

	#[test]
	fn unaligned_dims_are_rejected() {
		// The constructor's whole job beyond storing is to refuse a grid that would not
		// divide into whole 32×32 tiles.
		assert!(std::panic::catch_unwind(|| LayerGeometry::new(128, 5121, 17408)).is_err());
		assert!(std::panic::catch_unwind(|| LayerGeometry::new(128, 5120, 17409)).is_err());
	}

	#[test]
	fn aligned_dims_give_exact_grids() {
		for &(n_dim, n_ffn) in &[(1024u32, 3456), (2048, 6912), (4096, 13824), (5120, 17408)] {
			let g = LayerGeometry::new(128, n_dim, n_ffn);
			assert_eq!((g.n_dim * g.n_ffn) % (32 * 32), 0, "grid must divide for {n_dim}×{n_ffn}");
		}
	}

	#[test]
	fn ping_pong_alternates_roles() {
		let mut pp = PingPongActivation {
			buf_a: vk::Buffer::from_raw(1),
			buf_b: vk::Buffer::from_raw(2),
			active: 0,
			size_bytes: 4096,
		};

		// Start: A reads, B scrubs.
		assert_eq!(pp.read_buffer(), vk::Buffer::from_raw(1));
		assert_eq!(pp.scratch_buffer(), vk::Buffer::from_raw(2));

		// After each layer pass the roles must exchange, never collide.
		for _ in 0..8 {
			let (r, s) = (pp.read_buffer(), pp.scratch_buffer());
			assert_ne!(r, s, "read and scratch must be different buffers");
			pp.swap();
		}
		// Even number of swaps → back to the start orientation.
		assert_eq!(pp.read_buffer(), vk::Buffer::from_raw(1));
		assert_eq!(pp.scratch_buffer(), vk::Buffer::from_raw(2));
	}

	#[test]
	fn dual_binding_picks_up_current_roles() {
		let pp = PingPongActivation {
			buf_a: vk::Buffer::from_raw(1),
			buf_b: vk::Buffer::from_raw(2),
			active: 0,
			size_bytes: 4096,
		};
		let layout = DualBindingLayout::new(&pp, vk::Buffer::from_raw(3), 0);
		assert_eq!(layout.act_buffer, vk::Buffer::from_raw(1));
		assert_eq!(layout.scratch_buffer, vk::Buffer::from_raw(2));
		assert_eq!(layout.weight_region, vk::Buffer::from_raw(3));
	}
}

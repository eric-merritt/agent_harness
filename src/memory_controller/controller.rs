use crate::memory_controller::virtual_tensor_arena::{
	OperationType, PageResidency, VirtualTensorArena,
};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use std::sync::{Arc, Mutex, OnceLock};
use sysinfo::System;

/// A block of model data to be paged into the arena.
#[derive(Clone, Debug)]
pub struct BlockDescriptor {
	/// File/offset within the source model
	pub offset: u64,
	/// Number of bytes in this block
	pub size: u64,
}

/// Per-dispatch parameters for the sandbag quantize shader.
///
/// Every field is a byte or element offset into the single flat arena binding, so
/// one immutable descriptor set serves every tensor in the model. Layout must match
/// the `push_constant` block in `src/models/sandbag_quantize.comp` exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct QuantizePushConstants {
	/// Byte offset of this tensor's source weights within the arena.
	pub src_offset: u32,
	/// Byte offset of this tensor's per-block scale plane.
	pub scale_offset: u32,
	/// Byte offset of this tensor's prefix/tail plane.
	pub pairs_offset: u32,
	/// Byte offset of this tensor's sign plane (u64 words, at the very end).
	pub sign_offset: u32,
	/// Number of weights in this tensor.
	pub elem_count: u32,
	/// Source element type: 0 = F32, 1 = F16, 2 = BF16.
	pub src_type: u32,
	/// Tail digits retained: 0..=3.
	pub tail_digits: u32,
	/// Saturation point from CPU-side calibration.
	pub threshold: f32,
}

/// Per-dispatch parameters for the Hessian/Trellis sensitivity shader
/// (`hessian_trellis.comp`). One workgroup = one 32×32 tile, so these address a single
/// tile of the paged weight space plus the activation rows it reads. Layout must match the
/// `push_constant` block in that shader exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct HessianTrellisPushConstants {
	/// Byte offset of this tile's 32×32 f16 block inside the weight (binding 1) arena.
	pub tile_offset_bytes: u32,
	/// nDim — elements per activation row. (Reserved; the shader indexes rows directly.)
	pub act_row_stride: u32,
	/// Element offset of the activation rows this tile reads (token·nDim into binding 0).
	pub act_base_elem: u32,
	/// First input-dim column this tile covers (a multiple of 32).
	pub col0: u32,
	/// Index into the output plane (binding 2) for this tile's vec4 result.
	pub out_tile_index: u32,
	/// Rademacher iterations to average — more gives a tighter Hessian estimate.
	pub n_iters: u32,
	/// Philox base seed; made unique per tile so projections decorrelate across tiles.
	pub seed: u32,
}

// ── Task 1: Dual-Binding Sequential Memory Architecture ────────────────────────
//
// A host-side plan for the two descriptor bindings that drive one layer pass in the
// alternating execution loop. It is pure layout math — no Vulkan handles — so it can be
// constructed, inspected and unit-tested on the CPU before a device exists. The actual
// buffers are created once (pre-allocated) and these structs only name which pre-allocated
// block plays which role this pass, plus the exact byte sizes the allocator must honour.
//
// Nothing here is hardcoded to a model shape: every size is derived from the layer's
// (tokens, nDim_in, nDim_out) triple through the formulas in the spec, so the same code
// serves both square Attention layers and wide SwiGLU FFN expansions.

/// The two physical blocks behind binding 0. Identical in size; they swap roles every
/// layer pass (ping-pong). Block A is "read" this pass while block B accumulates, then
/// the barrier at end-of-pass promotes B's scratch into next pass's read-only input and
/// the roles flip. `active` is which block is the READ-ONLY INPUT this pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PingPongRole {
	/// This block holds the layer's input activations (read-only) for this pass.
	ReadInput,
	/// This block is the writable scratch accumulator (zeroed at pass start, atomicAdd).
	WriteScratch,
}

/// Per-pass view of the ping-pong activation space behind binding 0.
#[derive(Clone, Copy, Debug)]
pub struct PingPongActivation {
	/// Bytes in ONE activation block: tokens × nDim × 4 (FP32). Both blocks are this size.
	pub block_bytes: u64,
	/// Which physical block (0 = A, 1 = B) is the read-only input THIS pass.
	pub active_read_block: usize,
}

impl PingPongActivation {
	/// Size formula: Sequence Length (Tokens) × Hidden Dimension (nDim) × 4 bytes (FP32).
	///
	/// For a 5,120-wide hidden dimension at 4,096 tokens this is exactly 83,886,080 bytes
	/// (80.00 MiB). Both pre-allocated blocks are this size.
	pub fn new(tokens: u64, n_dim: u64) -> Self {
		let block_bytes = tokens * n_dim * 4;
		Self {
			block_bytes,
			active_read_block: 0,
		}
	}

	/// The read-only input block's byte range within the activation arena.
	pub fn read_input_range(&self) -> (u64, u64) {
		(self.active_read_block as u64 * self.block_bytes, self.block_bytes)
	}

	/// The writable scratch block's byte range — the one NOT being read this pass.
	pub fn write_scratch_range(&self) -> (u64, u64) {
		let scratch_block = 1 - self.active_read_block;
		(scratch_block as u64 * self.block_bytes, self.block_bytes)
	}

	/// Flip roles for the next layer pass: this pass's scratch becomes next pass's input.
	pub fn advance(&mut self) {
		self.active_read_block = 1 - self.active_read_block;
	}
}

/// A single weight tensor page inside the paged sparse tile space (binding 1).
#[derive(Clone, Copy, Debug)]
pub struct WeightTilePage {
	/// Input dimension (nDim_in) of this tensor. Must be a multiple of 32 for tiling.
	pub n_dim_in: u64,
	/// Output dimension (nDim_out) of this tensor. Multiple of 32; may exceed n_dim_in
	/// for wide FFN expansion layers (SwiGLU up/gate).
	pub n_dim_out: u64,
	/// Byte offset of this page's first tile within the weight arena.
	pub offset_bytes: u64,
}

impl WeightTilePage {
	/// Sizing formula: nDim_in × nDim_out × 2 bytes (FP16).
	pub fn size_bytes(&self) -> u64 {
		self.n_dim_in * self.n_dim_out * 2
	}

	/// The tile grid. Each tile is 32×32 elements; the whole tensor must tile exactly, so
	/// both dims are required to be multiples of 32 (true for square attention and for the
	/// 5120→17408 SwiGLU expansion alike).
	pub fn tile_grid(&self) -> (u64, u64) {
		(self.n_dim_in / 32, self.n_dim_out / 32) // (horizontal, vertical) tiles
	}

	/// Number of 2 KB (32×32 FP16) tiles in this page.
	pub fn tile_count(&self) -> u64 {
		let (h, v) = self.tile_grid();
		h * v
	}

	/// Bytes per tile: 32 × 32 elements × 2 bytes (FP16) = 2048.
	pub const TILE_BYTES: u64 = 32 * 32 * 2;
}

/// The paged, read-only FP16 weight tile space behind binding 1.
///
/// Weights are laid out as a continuous linear run of 32×32 FP16 tiles, entirely separate
/// from the activation arena so each stays cache-aligned on its own alignment boundary.
#[derive(Clone, Copy, Debug)]
pub struct WeightTileSpace {
	/// Total bytes reserved for every weight page in this space.
	pub total_bytes: u64,
	/// One 32×32 FP16 tile is always exactly this many bytes.
	pub tile_bytes: u64,
}

impl WeightTileSpace {
	/// Build a paged weight space from a list of tensor shapes (one per layer tensor).
	///
	/// Pages are packed back-to-back with no interior padding, so page *n*'s offset equals
	/// the sum of every earlier page's size — the same running-sum invariant the arena paging
	/// relies on. Square attention (5120×5120) and wide FFN (5120×17408) both tile exactly.
	pub fn new(tensor_shapes: &[(u64, u64)]) -> Self {
		let mut total_bytes = 0u64;
		for &(n_dim_in, n_dim_out) in tensor_shapes {
			debug_assert_eq!(n_dim_in % 32, 0, "nDim_in must tile by 32");
			debug_assert_eq!(n_dim_out % 32, 0, "nDim_out must tile by 32");
			total_bytes += n_dim_in * n_dim_out * 2;
		}
		Self {
			total_bytes,
			tile_bytes: WeightTilePage::TILE_BYTES,
		}
	}

	/// Lay out `tensor_shapes` into consecutive pages. Returns each page's offset so the
	/// caller can hand the running-sum layout to the upload path.
	pub fn plan_pages(&self, tensor_shapes: &[(u64, u64)]) -> Vec<WeightTilePage> {
		let mut pages = Vec::with_capacity(tensor_shapes.len());
		let mut offset = 0u64;
		for &(n_dim_in, n_dim_out) in tensor_shapes {
			pages.push(WeightTilePage {
				n_dim_in,
				n_dim_out,
				offset_bytes: offset,
			});
			offset += n_dim_in * n_dim_out * 2;
		}
		pages
	}
}

/// The full per-pass descriptor layout for one layer of the alternating loop.
#[derive(Clone, Copy, Debug)]
pub struct LayerBindingLayout {
	/// binding = 0: ping-pong activation space (input read + scratch write).
	pub activations: PingPongActivation,
	/// binding = 1: this layer's weight tensor page inside the paged tile space.
	pub weights: WeightTilePage,
}

impl LayerBindingLayout {
	/// Assemble a per-pass layout from the shared activation space (sized once for the
	/// whole model by its max hidden dim and token budget) and this layer's weight shape.
	pub fn new(activations: PingPongActivation, n_dim_in: u64, n_dim_out: u64, offset_bytes: u64) -> Self {
		Self {
			activations,
			weights: WeightTilePage {
				n_dim_in,
				n_dim_out,
				offset_bytes,
			},
		}
	}

	/// The FFN intermediate dimension for a SwiGLU layer, rounded up to the next multiple of
	/// 32 so the expansion tiles exactly.
	///
	/// NOTE: this is taken from the model's own config (`intermediate_size`), NOT derived
	/// from the hidden dim — Qwen checkpoints pick it independently and the ratio is not a
	/// clean constant (for the 5,120-hidden baseline the real value is 17,408, i.e. ×3.4,
	/// which is *not* 8/3 despite what older notes claim). The caller passes the checkpoint's
	/// actual number; this only guarantees it tiles by 32 and reports it back.
	pub fn swiglu_intermediate(inter_dim: u64) -> u64 {
		(inter_dim + 31) / 32 * 32
	}
}

// GPU context — holds Vulkan device, queue, allocator, and command pool handles.
pub struct GpuContext {
	pub device_handle: ash::Device,
	pub physical_device: vk::PhysicalDevice,
	pub queue_handle: vk::Queue,
	pub queue_family: u32,
	pub allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
	pub command_pool: vk::CommandPool,
	/// Pooled command buffers — recycled via resetCommandBuffers instead of re-allocated.
	pub cmd_buffer_pool: std::sync::Mutex<Vec<vk::CommandBuffer>>,
	/// Pooled fences — recycled via resetFences instead of create/destroy.
	pub fence_pool: std::sync::Mutex<Vec<vk::Fence>>,
	/// Cached quantization compute pipeline (compiled from quantize_gemv.spv).
	pub cached_quantize_pipeline: vk::Pipeline,
	/// Cached pipeline layout (push constants + descriptor set).
	pub cached_pipeline_layout: vk::PipelineLayout,
	/// Descriptor-set layout (needed to allocate/update descriptor sets dynamically).
	pub cached_descriptor_set_layout: vk::DescriptorSetLayout,
	/// Descriptor pool (needed to allocate new sets).
	pub cached_descriptor_pool: vk::DescriptorPool,
	/// Cached descriptor set binding the sparse buffer to the shader.
	pub cached_descriptor_set: vk::DescriptorSet,
}

impl GpuContext {
	pub fn new(
		device: ash::Device,
		physical_device: vk::PhysicalDevice,
		queue: vk::Queue,
		queue_family: u32,
		allocator: Arc<Mutex<Allocator>>,
		command_pool: vk::CommandPool,
		cached_quantize_pipeline: vk::Pipeline,
		cached_pipeline_layout: vk::PipelineLayout,
		cached_descriptor_set_layout: vk::DescriptorSetLayout,
		cached_descriptor_pool: vk::DescriptorPool,
		cached_descriptor_set: vk::DescriptorSet,
	) -> Self {
		Self {
			device_handle: device,
			physical_device,
			queue_handle: queue,
			queue_family,
			allocator,
			command_pool,
			cmd_buffer_pool: std::sync::Mutex::new(Vec::new()),
			fence_pool: std::sync::Mutex::new(Vec::new()),
			cached_quantize_pipeline,
			cached_pipeline_layout,
			cached_descriptor_set_layout,
			cached_descriptor_pool,
			cached_descriptor_set,
		}
	}

	/// Shallow clone for parallel worker contexts — shares Arc handles.
	pub fn clone_shallow(&self) -> Self {
		Self {
			device_handle: self.device_handle.clone(),
			physical_device: self.physical_device,
			queue_handle: self.queue_handle,
			queue_family: self.queue_family,
			allocator: Arc::clone(&self.allocator),
			command_pool: self.command_pool,
			cmd_buffer_pool: std::sync::Mutex::new(Vec::new()),
			fence_pool: std::sync::Mutex::new(Vec::new()),
			cached_quantize_pipeline: self.cached_quantize_pipeline,
			cached_pipeline_layout: self.cached_pipeline_layout,
			cached_descriptor_set_layout: self.cached_descriptor_set_layout,
			cached_descriptor_pool: self.cached_descriptor_pool,
			cached_descriptor_set: self.cached_descriptor_set,
		}
	}
	pub fn device(&self) -> &ash::Device {
		&self.device_handle
	}
	pub fn queue(&self) -> vk::Queue {
		self.queue_handle
	}
	pub fn allocator(&self) -> Arc<Mutex<gpu_allocator::vulkan::Allocator>> {
		Arc::clone(&self.allocator)
	}

	// ── Command-buffer / fence pool helpers ──

	/// Pop a recycled command buffer from the pool, or allocate a fresh one.
	pub fn alloc_cmd_buffer(
		device: &ash::Device,
		pool: vk::CommandPool,
		pooled: &std::sync::Mutex<Vec<vk::CommandBuffer>>,
	) -> vk::CommandBuffer {
		let mut guard = pooled.lock().unwrap();
		let cmd = guard.pop().unwrap_or_else(|| {
			drop(guard);
			let alloc_info = vk::CommandBufferAllocateInfo::default()
				.command_pool(pool)
				.level(vk::CommandBufferLevel::PRIMARY)
				.command_buffer_count(1);
			unsafe {
				device
					.allocate_command_buffers(&alloc_info)
					.expect("allocate command buffer")[0]
			}
		});
		cmd
	}

	/// Pop a recycled fence from the pool, or create a fresh one.
	pub fn alloc_fence(
		device: &ash::Device,
		pooled: &std::sync::Mutex<Vec<vk::Fence>>,
	) -> vk::Fence {
		let mut guard = pooled.lock().unwrap();
		guard.pop().unwrap_or_else(|| {
			drop(guard);
			unsafe {
				device
					.create_fence(&vk::FenceCreateInfo::default(), None)
					.expect("create fence")
			}
		})
	}

	/// Recycle a waited-on fence: reset and push back into the pool.
	pub fn recycle_fence(
		device: &ash::Device,
		fence: vk::Fence,
		pooled: &std::sync::Mutex<Vec<vk::Fence>>,
	) {
		unsafe {
			device.reset_fences(&[fence]).expect("reset fence for pool");
		}
		pooled.lock().unwrap().push(fence);
	}

	/// Recycle an executed command buffer: reset and push back into the pool.
	pub fn recycle_cmd_buffer(
		device: &ash::Device,
		cmd: vk::CommandBuffer,
		pooled: &std::sync::Mutex<Vec<vk::CommandBuffer>>,
	) {
		unsafe {
			let result = device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty());
			match result {
				Ok(_) => {
					println!("Successfully recycled command buffer.");
				},
				Err(_) => {
					eprintln!("Failed to recycle command buffer.");
				}
			}
		}
		pooled.lock().unwrap().push(cmd);
	}

	/// Synchronous upload: copy `data` into `buf` at `offset` via a staging buffer.
	///
	/// Deprecated — kept for callers that still need one-off uploads.
	/// For batched page writes, use `batch_upload` instead.
	pub unsafe fn upload(&self, buf: vk::Buffer, offset: vk::DeviceSize, data: &[u8]) {
		use std::time::Instant;
		let t = Instant::now();
		let size = data.len() as vk::DeviceSize;
		eprintln!(
			"[UPLOAD] t=0ms  START — {} bytes to buf={:?} offset={}",
			size, buf, offset
		);
		if size == 0 {
			eprintln!(
				"[UPLOAD] t+{:3}ms  size==0, returning early",
				t.elapsed().as_millis()
			);
			return;
		}

		// 1. Create staging buffer
		let staging = unsafe {
			self.device_handle
				.create_buffer(
					&vk::BufferCreateInfo::default()
						.size(size)
						.usage(vk::BufferUsageFlags::TRANSFER_SRC),
					None,
				)
				.expect("create staging buffer")
		};
		let mem_reqs = unsafe { self.device_handle.get_buffer_memory_requirements(staging) };

		// 2. Allocate host-visible memory
		let mut guard = self.allocator.lock().unwrap();
		let alloc = guard
			.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
				name: "staging_upload",
				requirements: mem_reqs,
				location: gpu_allocator::MemoryLocation::CpuToGpu,
				linear: true,
				allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
			})
			.expect("allocate staging memory");
		drop(guard);

		unsafe {
			self.device_handle
				.bind_buffer_memory(staging, alloc.memory(), alloc.offset())
				.expect("bind staging buffer");
		}

		// 3. Copy data into mapped staging memory
		if let Some(ptr) = alloc.mapped_ptr() {
			unsafe {
				std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>().as_ptr(), data.len());
			}
		} else {
			panic!("Staging allocation is not host-mapped");
		}

		// 4. Record + submit in one batch
		let copies = vec![vk::BufferCopy::default()
			.src_offset(0)
			.dst_offset(offset)
			.size(size)];

		unsafe {
			self.submit_copy_batch(staging, buf, &copies, t);
		}

		// 5. Cleanup staging buffer + allocation
		unsafe {
			self.device_handle.destroy_buffer(staging, None);
		}
		let mut guard = self.allocator.lock().unwrap();
		let _ = guard.free(alloc);
		eprintln!(
			"[UPLOAD] t+{:3}ms  DONE — {} bytes",
			t.elapsed().as_millis(),
			data.len()
		);
	}

	/// Batch multiple buffer copies into a single staging buffer + one submit.
	/// Each `(staging_offset, dst_offset, size)` describes one copy region.
	pub unsafe fn batch_upload(
		&self,
		buf: vk::Buffer,
		regions: &[(vk::DeviceSize, vk::DeviceSize, vk::DeviceSize)],
		data: &[&[u8]],
	) {
		use std::time::Instant;
		let t = Instant::now();

		if regions.is_empty() || data.is_empty() {
			return;
		}

		// Compute total staging size
		let total_size: vk::DeviceSize = regions.iter().map(|r| r.2).sum();
		let mut staging_data = Vec::with_capacity(total_size as usize);
		for (i, (_, _, _)) in regions.iter().enumerate() {
			staging_data.extend_from_slice(data[i]);
		}

		// Create single staging buffer for all copies
		let staging = unsafe {
			self.device_handle
				.create_buffer(
					&vk::BufferCreateInfo::default()
						.size(total_size)
						.usage(vk::BufferUsageFlags::TRANSFER_SRC),
					None,
				)
				.expect("create batch staging buffer")
		};
		let mem_reqs = unsafe { self.device_handle.get_buffer_memory_requirements(staging) };

		let mut guard = self.allocator.lock().unwrap();
		let alloc = guard
			.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
				name: "batch_staging_upload",
				requirements: mem_reqs,
				location: gpu_allocator::MemoryLocation::CpuToGpu,
				linear: true,
				allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
			})
			.expect("allocate batch staging memory");
		drop(guard);

		unsafe {
			self.device_handle
				.bind_buffer_memory(staging, alloc.memory(), alloc.offset())
				.expect("bind batch staging buffer");
		}

		// Write all data into mapped memory
		if let Some(ptr) = alloc.mapped_ptr() {
			unsafe {
				std::ptr::copy_nonoverlapping(
					staging_data.as_ptr(),
					ptr.cast::<u8>().as_ptr(),
					staging_data.len(),
				);
			}
		} else {
			panic!("Batch staging allocation is not host-mapped");
		}

		// Build copy regions — use the running staging offset so each page
		// reads from its own slice, not all from zero.
		let mut copies = Vec::with_capacity(regions.len());
		for (stg_off, dst_off, sz) in regions {
			copies.push(vk::BufferCopy::default()
				.src_offset(*stg_off)
				.dst_offset(*dst_off)
				.size(*sz));
		}

		unsafe {
			self.submit_copy_batch(staging, buf, &copies, t);
		}

		// Cleanup
		unsafe {
			self.device_handle.destroy_buffer(staging, None);
		}
		let mut guard = self.allocator.lock().unwrap();
		let _ = guard.free(alloc);
	}

	/// Shared submit: record copies, submit, fence, recycle.
	unsafe fn submit_copy_batch(
		&self,
		staging: vk::Buffer,
		dst_buf: vk::Buffer,
		copies: &[vk::BufferCopy],
		_t: std::time::Instant,
	) {
		// Queue idle ensures no other work is in-flight on this queue
		unsafe {
			self.device_handle
				.queue_wait_idle(self.queue_handle)
				.expect("queue_wait_idle before cmd alloc");
		}

		let cmd = Self::alloc_cmd_buffer(
			&self.device_handle,
			self.command_pool,
			&self.cmd_buffer_pool,
		);
		unsafe {
			self.device_handle
				.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
				.expect("begin command buffer");
			self.device_handle.cmd_copy_buffer(
				cmd,
				staging,
				dst_buf,
				copies,
			);
			self.device_handle
				.end_command_buffer(cmd)
				.expect("end command buffer");
		}

		// Submit and wait
		let fence = Self::alloc_fence(&self.device_handle, &self.fence_pool);
		unsafe {
			self.device_handle
				.queue_submit(
					self.queue_handle,
					&[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
					fence,
				)
				.expect("submit upload batch");
			self.device_handle
				.wait_for_fences(&[fence], true, u64::MAX)
				.expect("wait upload fence");
		}

		// Recycle fence and command buffer
		unsafe {
			Self::recycle_fence(&self.device_handle, fence, &self.fence_pool);
			let result = self.device_handle.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty());
			match result {
				Ok(_) => {
					println!("Successfully recycled fence and command buffer. Pushing.");
					self.cmd_buffer_pool.lock().unwrap().push(cmd);
				},
				Err(_) => {
					eprintln!("Failed to recycle command buffer. Attempting recovery.");
				}
			}
		}

	}

	/// Synchronous download: copy `size` bytes from `buf` at `offset` into a Vec.
	pub unsafe fn download(
		&self,
		buf: vk::Buffer,
		offset: vk::DeviceSize,
		size: vk::DeviceSize,
	) -> Vec<u8> {
		use std::time::Instant;
		let t = Instant::now();
		eprintln!(
			"[DOWNLOAD] t=0ms  START — {} bytes from buf={:?} offset={}",
			size, buf, offset
		);
		if size == 0 {
			eprintln!("[DOWNLOAD] size==0, returning empty vec");
			return Vec::new();
		}

		// 1. Create staging buffer
		eprintln!(
			"[DOWNLOAD] t+{:3}ms  creating staging buffer...",
			t.elapsed().as_millis()
		);

		// 1. Create staging buffer
		let staging = unsafe {
			self.device_handle
				.create_buffer(
					&vk::BufferCreateInfo::default()
						.size(size)
						.usage(vk::BufferUsageFlags::TRANSFER_DST),
					None,
				)
				.expect("create staging buffer")
		};
		let mem_reqs = unsafe { self.device_handle.get_buffer_memory_requirements(staging) };

		// 2. Allocate host-visible memory
		let mut guard = self.allocator.lock().unwrap();
		let alloc = guard
			.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
				name: "staging_download",
				requirements: mem_reqs,
				location: gpu_allocator::MemoryLocation::GpuToCpu,
				linear: true,
				allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
			})
			.expect("allocate staging memory");
		drop(guard);

		unsafe {
			self.device_handle
				.bind_buffer_memory(staging, alloc.memory(), alloc.offset())
				.expect("bind staging buffer");
		}

		// 3. Record copy command from pooled buffer
		let cmd = Self::alloc_cmd_buffer(
			&self.device_handle,
			self.command_pool,
			&self.cmd_buffer_pool,
		);
		unsafe {
			self.device_handle
				.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
				.expect("begin command buffer");
			self.device_handle.cmd_copy_buffer(
				cmd,
				buf,
				staging,
				&[vk::BufferCopy::default()
					.src_offset(offset)
					.dst_offset(0)
					.size(size)],
			);
			self.device_handle
				.end_command_buffer(cmd)
				.expect("end command buffer");
		}

		// 4. Submit and wait (reuse fence from pool)
		eprintln!(
			"[DOWNLOAD] t+{:3}ms  creating fence + submitting...",
			t.elapsed().as_millis()
		);
		let fence = Self::alloc_fence(&self.device_handle, &self.fence_pool);
		unsafe {
			self.device_handle
				.queue_submit(
					self.queue_handle,
					&[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
					fence,
				)
				.expect("submit download");
			eprintln!(
				"t+{:3}ms  queue_submit done, waiting on fence (BLOCKS HERE = GPU HANG)...",
				t.elapsed().as_millis()
			);
			self.device_handle
				.wait_for_fences(&[fence], true, u64::MAX)
				.expect("wait download fence");
			eprintln!(
				"[DOWNLOAD] t+{:3}ms  fence signaled, reading data...",
				t.elapsed().as_millis()
			);
		}

		// 5. Read data from mapped staging memory
		let result_size = alloc.size() as usize;
		let mut result = vec![0u8; result_size];
		if let Some(ptr) = alloc.mapped_ptr() {
			unsafe {
				std::ptr::copy_nonoverlapping(
					ptr.cast::<u8>().as_ptr(),
					result.as_mut_ptr(),
					result_size,
				);
			}
		}

		// 6. Cleanup — recycle cmd buffer + fence back to pools
		unsafe {
			Self::recycle_fence(&self.device_handle, fence, &self.fence_pool);
			self.device_handle
				.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty());
			self.cmd_buffer_pool.lock().unwrap().push(cmd);
			self.device_handle.destroy_buffer(staging, None);
		}
		let mut guard = self.allocator.lock().unwrap();
		let _ = guard.free(alloc);

		result
	}
}

/// CPU memory backing pool — wraps the cpu_mem_op pool.
pub struct CpuMemoryManager {
	pool: crate::memory_controller::cpu_mem_op::CpuMemory,
}

impl CpuMemoryManager {
	/// Instantiates the manager by dynamically querying the operating system
	/// and allocating the anonymous virtual backing memory mapping.
	pub fn new() -> Self {
		let cpu_avail_mem = crate::memory_controller::cpu_mem_op::CpuMemory::get_avail_cpu_mem();
		let reserve = 4_000_000_000 as usize; // Keep 4 GB host operating system headroom

		// Ensure we don't underflow if the environment is heavily resource-constrained
		let total_size = cpu_avail_mem.saturating_sub(reserve);

		// Allocate the memory map and unwrap the initialization Result safely
		let pool = crate::memory_controller::cpu_mem_op::CpuMemory::new(total_size)
			.expect("Failed to initialize anonymous virtual memory mapping for CPU backing pool");

		Self { pool }
	}

	pub fn capacity(&self) -> usize {
		self.pool.capacity()
	}

	pub fn write_page(&mut self, idx: usize, size: usize, data: &[u8]) {
		self.pool.write_page(idx, size, data);
	}

	pub fn read_page(&self, idx: usize, size: usize) -> &[u8] {
		self.pool.read_page(idx, size)
	}

	pub fn drop_page(&mut self, idx: usize, size: usize) {
		self.pool.drop_page(idx, size);
	}
}

pub struct MemoryController {
	pub arena: VirtualTensorArena,
	pub gpu: GpuContext,
	pub cpu: CpuMemoryManager,
	pub max_cpu_bytes: u64,
	pub used_cpu_bytes: u64,
	pub max_vram_bytes: u64,
	pub used_vram_bytes: u64,
}

impl MemoryController {
	pub fn cpu_available(&self) -> u64 {
		self.max_cpu_bytes.saturating_sub(self.used_cpu_bytes)
	}

	pub fn vram_available(&self) -> u64 {
		self.max_vram_bytes.saturating_sub(self.used_vram_bytes)
	}

	pub fn free_cpu_space(&mut self, _bytes: u64) {
		// Implementation for cleaning or paging out cold RAM blocks
	}

	/// Free VRAM by evicting GPU-resident pages to CPU.
	/// Ensures CPU RAM is available for the incoming data first.
	pub fn free_vram_space(&mut self, bytes: u64, exclude_page: usize) -> u64 {
		let page_size = self.arena.page_size as usize;
		let mut freed = 0u64;

		for page_index in 0..self.arena.total_pages {
			if freed >= bytes {
				break;
			}
			if page_index == exclude_page {
				continue;
			}

			let residency = self.arena.page_table[page_index].residency;
			if residency != PageResidency::GpuResident {
				continue;
			}

			let needed = page_size as u64;
			if self.cpu_available() < needed {
				let deficit = needed - self.cpu_available();
				self.free_cpu_space(deficit);
			}

			let data = self.download_page(page_index);
			self.cpu.write_page(page_index, page_size, &data);
			self.evict_page(page_index);

			let page = &mut self.arena.page_table[page_index];
			page.residency = PageResidency::CpuResident;
			page.cpu_offset = Some(page_index * page_size);

			freed += page_size as u64;
		}

		if freed < bytes {
			log::warn!(
				"free_vram_space: needed {} bytes, only freed {} — VRAM may be exhausted",
				bytes,
				freed
			);
		}
		freed
	}
	// ── Page operations ────────────────────────────────────────────────

	/// Commit a page to GPU VRAM. Falls back to CPU on OOM.
	pub fn commit_page(&mut self, page_index: usize) {
		unsafe {
			self.arena
				.commit_page(self.gpu.device(), self.gpu.queue(), page_index);
		}
	}

	/// Commit multiple pages. Serial — each sparse bind must be sequential.
	pub fn commit_pages(&mut self, page_indices: &[usize]) {
		for &page_idx in page_indices {
			if self.arena.page_table[page_idx].residency == PageResidency::Unmapped {
				self.commit_page(page_idx);
			}
		}
	}

	/// Evict a page from GPU — unbind and free VRAM. Data is NOT preserved.
	pub fn evict_page(&mut self, page_index: usize) {
		let op_type = OperationType::Drop;

		unsafe {
			self.arena.evict_page(
				page_index,
				self.gpu.allocator(),
				self.gpu.queue(),
				self.gpu.device(),
				op_type,
			);
		}
	}

	/// Upload data to a GPU page. Commits the page first if needed, then uploads.
	/// After successful upload, drops the CPU copy to free physical RAM.
	/// If VRAM is full, evicts a cold GPU page to make space.
	pub fn upload_page(&mut self, page_index: usize, data: &[u8]) {
		let page_size = self.arena.page_size as usize;

		let residency = self.arena.page_table[page_index].residency;
		if residency == PageResidency::Unmapped {
			let needed = page_size as u64;
			if self.vram_available() < needed {
				self.free_vram_space(needed, page_index);
			}
			self.commit_page(page_index);
		}

		let residency = self.arena.page_table[page_index].residency;
		if residency != PageResidency::GpuResident {
			log::warn!(
				"upload_page: page {} not GPU-resident after commit, keeping on CPU",
				page_index
			);
			self.cpu.write_page(page_index, page_size, data);
			let page = &mut self.arena.page_table[page_index];
			page.residency = PageResidency::CpuResident;
			page.cpu_offset = Some(page_index * page_size);
			return;
		}

		let offset = page_index as vk::DeviceSize * self.arena.page_size;
		unsafe {
			self.gpu.upload(self.arena.sparse_buffer, offset, data);
		}

		self.cpu.drop_page(page_index, page_size);
	}

	/// Download a page's data from GPU. Page must be GPU-resident.
	pub fn download_page(&self, page_index: usize) -> Vec<u8> {
		let offset = page_index as vk::DeviceSize * self.arena.page_size;
		let size = self.arena.page_size;
		unsafe { self.gpu.download(self.arena.sparse_buffer, offset, size) }
	}
	/// Evict a GPU page but preserve its data in CPU memory.
	/// Ensures CPU RAM is available before downloading.
	pub fn evict_page_with_data(&mut self, page_index: usize) {
		let page_size = self.arena.page_size as usize;

		let needed = page_size as u64;
		if self.cpu_available() < needed {
			let deficit = needed - self.cpu_available();
			self.free_cpu_space(deficit);
		}

		let data = self.download_page(page_index);
		self.cpu.write_page(page_index, page_size, &data);
		self.evict_page(page_index);

		let page = &mut self.arena.page_table[page_index];
		page.residency = PageResidency::CpuResident;
		page.cpu_offset = Some(page_index * page_size);
	}

	/// Promote a CPU-resident page to GPU, freeing its CPU RAM.
	pub fn migrate_to_gpu(&mut self, page_index: usize) {
		let page_size = self.arena.page_size as usize;
		let residency = self.arena.page_table[page_index].residency;
		if residency != PageResidency::CpuResident {
			return;
		}

		let data = self.cpu.read_page(page_index, page_size).to_vec();

		let page = &mut self.arena.page_table[page_index];
		page.residency = PageResidency::Unmapped;
		page.cpu_offset = None;

		self.upload_page(page_index, &data);
	}

	/// Demote a GPU-resident page to CPU, preserving data.
	pub fn migrate_to_cpu(&mut self, page_index: usize) {
		let residency = self.arena.page_table[page_index].residency;
		if residency != PageResidency::GpuResident {
			return;
		}
		self.evict_page_with_data(page_index);
	}

	/// Read a page's data from wherever it lives. Returns owned bytes.
	pub fn read_page(&self, page_index: usize) -> Vec<u8> {
		let page_size = self.arena.page_size as usize;
		let residency = self.arena.page_table[page_index].residency;
		match residency {
			PageResidency::CpuResident => self.cpu.read_page(page_index, page_size).to_vec(),
			PageResidency::GpuResident => self.download_page(page_index),
			PageResidency::Unmapped => {
				panic!("Cannot read unmapped page {}", page_index);
			}
		}
	}

	/// Write data to a page. Routes to CPU or GPU based on current residency.
	/// If unmapped, writes to CPU and marks it CPU-resident.
	pub fn write_page(&mut self, page_index: usize, data: &[u8]) {
		let page_size = self.arena.page_size as usize;
		let residency = self.arena.page_table[page_index].residency;
		match residency {
			PageResidency::GpuResident => {
				let offset = page_index as vk::DeviceSize * self.arena.page_size;
				unsafe {
					self.gpu.upload(self.arena.sparse_buffer, offset, data);
				}
			}
			PageResidency::CpuResident | PageResidency::Unmapped => {
				self.cpu.write_page(page_index, page_size, data);
				let page = &mut self.arena.page_table[page_index];
				page.residency = PageResidency::CpuResident;
				page.cpu_offset = Some(page_index * page_size);
			}
		}
	}

	/// Place a page on GPU or CPU based on the target hint.
	/// On GPU: ensures VRAM space, commits, uploads, drops CPU copy.
	/// On CPU: ensures RAM space, writes, marks CPU-resident.
	pub fn place_page(&mut self, page_index: usize, data: &[u8], on_gpu: bool) {
		if on_gpu {
			self.upload_page(page_index, data);
		} else {
			let page_size = self.arena.page_size as usize;
			let needed = page_size as u64;
			if self.cpu_available() < needed {
				let deficit = needed - self.cpu_available();
				self.free_cpu_space(deficit);
			}
			self.cpu.write_page(page_index, page_size, data);
			let page = &mut self.arena.page_table[page_index];
			page.residency = PageResidency::CpuResident;
			page.cpu_offset = Some(page_index * page_size);
		}
	}

	/// GPU buffer handle + offset for shader binding of a GPU-resident page.
	pub fn gpu_binding(&self, page_index: usize) -> (vk::Buffer, vk::DeviceSize) {
		let offset = page_index as vk::DeviceSize * self.arena.page_size;
		(self.arena.sparse_buffer, offset)
	}

	/// Locate the compiled sandbag quantize shader.
	///
	/// Searched at runtime rather than `include_bytes!` so the crate still builds
	/// before `glslangValidator` has produced the artifact. Returns `None` if the
	/// shader has not been compiled yet — the GPU path then reports itself as
	/// unavailable instead of silently producing garbage.
	fn find_quantize_spirv() -> Option<Vec<u8>> {
		let mut candidates: Vec<std::path::PathBuf> = vec![
			std::path::PathBuf::from(concat!(
				env!("CARGO_MANIFEST_DIR"),
				"/src/models/sandbag_quantize.spv"
			)),
			std::path::PathBuf::from("src/models/sandbag_quantize.spv"),
		];
		if let Ok(exe) = std::env::current_exe() {
			if let Some(dir) = exe.parent() {
				candidates.push(dir.join("sandbag_quantize.spv"));
			}
		}

		for path in &candidates {
			if let Ok(bytes) = std::fs::read(path) {
				eprintln!("[CONTROLLER] loaded quantize shader from {}", path.display());
				return Some(bytes);
			}
		}
		None
	}

	/// Build the compute pipeline that quantizes weights in place inside the arena.
	///
	/// One storage-buffer binding covers the whole sparse arena — source weights and
	/// destination bytes are both addressed through it, so the shader sees a single
	/// flat address space and needs no staging hop. Per-dispatch parameters travel in
	/// push constants, which keeps the descriptor set immutable across every tensor.
	fn create_quantize_pipeline(
		device: &ash::Device,
		sparse_buffer: vk::Buffer,
		page_size: vk::DeviceSize,
		total_pages: usize,
	) -> (
		vk::Pipeline,
		vk::PipelineLayout,
		vk::DescriptorSetLayout,
		vk::DescriptorPool,
		vk::DescriptorSet,
	) {
		let binding = vk::DescriptorSetLayoutBinding::default()
			.binding(0)
			.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(1)
			.stage_flags(vk::ShaderStageFlags::COMPUTE);

		let set_layout = unsafe {
			device
				.create_descriptor_set_layout(
					&vk::DescriptorSetLayoutCreateInfo::default()
						.bindings(std::slice::from_ref(&binding)),
					None,
				)
				.expect("create quantize descriptor set layout")
		};

		let push_range = vk::PushConstantRange::default()
			.stage_flags(vk::ShaderStageFlags::COMPUTE)
			.offset(0)
			.size(std::mem::size_of::<QuantizePushConstants>() as u32);

		let pipeline_layout = unsafe {
			device
				.create_pipeline_layout(
					&vk::PipelineLayoutCreateInfo::default()
						.set_layouts(std::slice::from_ref(&set_layout))
						.push_constant_ranges(std::slice::from_ref(&push_range)),
					None,
				)
				.expect("create quantize pipeline layout")
		};

		let pool_size = vk::DescriptorPoolSize::default()
			.ty(vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(1);

		let descriptor_pool = unsafe {
			device
				.create_descriptor_pool(
					&vk::DescriptorPoolCreateInfo::default()
						.max_sets(1)
						.pool_sizes(std::slice::from_ref(&pool_size)),
					None,
				)
				.expect("create quantize descriptor pool")
		};

		let descriptor_set = unsafe {
			device
				.allocate_descriptor_sets(
					&vk::DescriptorSetAllocateInfo::default()
						.descriptor_pool(descriptor_pool)
						.set_layouts(std::slice::from_ref(&set_layout)),
				)
				.expect("allocate quantize descriptor set")[0]
		};

		// Bind the entire arena. The buffer is sparse, so this range is only backed
		// where pages have actually been committed.
		let arena_bytes = page_size * total_pages as vk::DeviceSize;
		let buffer_info = vk::DescriptorBufferInfo::default()
			.buffer(sparse_buffer)
			.offset(0)
			.range(if arena_bytes == 0 {
				vk::WHOLE_SIZE
			} else {
				arena_bytes
			});

		unsafe {
			device.update_descriptor_sets(
				&[vk::WriteDescriptorSet::default()
					.dst_set(descriptor_set)
					.dst_binding(0)
					.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
					.buffer_info(std::slice::from_ref(&buffer_info))],
				&[],
			);
		}

		// The shader artifact is optional at build time; without it the pipeline
		// handle stays null and quantize_gpu() refuses to dispatch.
		let spirv = match Self::find_quantize_spirv() {
			Some(bytes) => bytes,
			None => {
				eprintln!(
					"[CONTROLLER] sandbag_quantize.spv not found — GPU quantize disabled. \
					 Compile it with: glslangValidator --target-env vulkan1.3 -o \
					 src/models/sandbag_quantize.spv src/models/sandbag_quantize.comp"
				);
				return (
					vk::Pipeline::null(),
					pipeline_layout,
					set_layout,
					descriptor_pool,
					descriptor_set,
				);
			}
		};

		let words: Vec<u32> = spirv
			.chunks_exact(4)
			.map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect();

		let shader_module = unsafe {
			device
				.create_shader_module(
					&vk::ShaderModuleCreateInfo::default().code(&words),
					None,
				)
				.expect("create quantize shader module")
		};

		let entry_name = c"main";
		let stage = vk::PipelineShaderStageCreateInfo::default()
			.stage(vk::ShaderStageFlags::COMPUTE)
			.module(shader_module)
			.name(entry_name);

		let pipeline = unsafe {
			device
				.create_compute_pipelines(
					vk::PipelineCache::null(),
					&[vk::ComputePipelineCreateInfo::default()
						.stage(stage)
						.layout(pipeline_layout)],
					None,
				)
				.expect("create quantize compute pipeline")[0]
		};

		// The module is baked into the pipeline; the handle is no longer needed.
		unsafe { device.destroy_shader_module(shader_module, None) };

		(
			pipeline,
			pipeline_layout,
			set_layout,
			descriptor_pool,
			descriptor_set,
		)
	}

	/// Locate the compiled Hessian/Trellis sensitivity shader (runtime-loaded, optional).
	fn find_hessian_spirv() -> Option<Vec<u8>> {
		let mut candidates: Vec<std::path::PathBuf> = vec![
			std::path::PathBuf::from(concat!(
				env!("CARGO_MANIFEST_DIR"),
				"/src/models/hessian_trellis.spv"
			)),
			std::path::PathBuf::from("src/models/hessian_trellis.spv"),
		];
		if let Ok(exe) = std::env::current_exe() {
			if let Some(dir) = exe.parent() {
				candidates.push(dir.join("hessian_trellis.spv"));
			}
		}
		for path in &candidates {
			if let Ok(bytes) = std::fs::read(path) {
				eprintln!("[CONTROLLER] loaded hessian shader from {}", path.display());
				return Some(bytes);
			}
		}
		None
	}

	/// Build the dual-binding Hessian/Trellis compute pipeline.
	///
	/// Three storage-buffer bindings: 0 = ping-pong activation space (read), 1 = paged FP16
	/// weight tile space (read), 2 = per-tile output plane (write). Per-tile addressing goes
	/// in push constants so the descriptor set stays immutable and one dispatch grid sweeps
	/// every tile of a layer. Follows `create_quantize_pipeline`'s structure exactly.
	fn create_hessian_pipeline(
		device: &ash::Device,
		activation_buffer: vk::Buffer,
		weight_buffer: vk::Buffer,
		output_buffer: vk::Buffer,
	) -> (
		vk::Pipeline,
		vk::PipelineLayout,
		vk::DescriptorSetLayout,
		vk::DescriptorPool,
		vk::DescriptorSet,
	) {
		let bindings = [
			vk::DescriptorSetLayoutBinding::default()
				.binding(0)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
			vk::DescriptorSetLayoutBinding::default()
				.binding(1)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
			vk::DescriptorSetLayoutBinding::default()
				.binding(2)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
		];

		let set_layout = unsafe {
			device
				.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
				.expect("create hessian descriptor set layout")
		};

		let push_range = vk::PushConstantRange::default()
			.stage_flags(vk::ShaderStageFlags::COMPUTE)
			.offset(0)
			.size(std::mem::size_of::<HessianTrellisPushConstants>() as u32);

		let pipeline_layout = unsafe {
			device
				.create_pipeline_layout(
					&vk::PipelineLayoutCreateInfo::default()
						.set_layouts(std::slice::from_ref(&set_layout))
						.push_constant_ranges(std::slice::from_ref(&push_range)),
					None,
				)
				.expect("create hessian pipeline layout")
		};

		let pool_size = vk::DescriptorPoolSize::default()
			.ty(vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(3);

		let descriptor_pool = unsafe {
			device
				.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(std::slice::from_ref(&pool_size)), None)
				.expect("create hessian descriptor pool")
		};

		let descriptor_set = unsafe {
			device
				.allocate_descriptor_sets(
					&vk::DescriptorSetAllocateInfo::default()
						.descriptor_pool(descriptor_pool)
						.set_layouts(std::slice::from_ref(&set_layout)),
				)
				.expect("allocate hessian descriptor set")[0]
		};

		let act_info = vk::DescriptorBufferInfo::default().buffer(activation_buffer).offset(0).range(vk::WHOLE_SIZE);
		let wgt_info = vk::DescriptorBufferInfo::default().buffer(weight_buffer).offset(0).range(vk::WHOLE_SIZE);
		let out_info = vk::DescriptorBufferInfo::default().buffer(output_buffer).offset(0).range(vk::WHOLE_SIZE);

		unsafe {
			device.update_descriptor_sets(
				&[
					vk::WriteDescriptorSet::default()
						.dst_set(descriptor_set)
						.dst_binding(0)
						.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
						.buffer_info(std::slice::from_ref(&act_info)),
					vk::WriteDescriptorSet::default()
						.dst_set(descriptor_set)
						.dst_binding(1)
						.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
						.buffer_info(std::slice::from_ref(&wgt_info)),
					vk::WriteDescriptorSet::default()
						.dst_set(descriptor_set)
						.dst_binding(2)
						.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
						.buffer_info(std::slice::from_ref(&out_info)),
				],
				&[],
			);
		}

		let spirv = match Self::find_hessian_spirv() {
			Some(bytes) => bytes,
			None => {
				eprintln!(
					"[CONTROLLER] hessian_trellis.spv not found — Hessian/Trellis pipeline disabled. \
					 Compile it with: glslangValidator --target-env vulkan1.3 -o \
					 src/models/hessian_trellis.spv src/models/hessian_trellis.comp"
				);
				return (vk::Pipeline::null(), pipeline_layout, set_layout, descriptor_pool, descriptor_set);
			}
		};

		let words: Vec<u32> = spirv.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

		let shader_module = unsafe {
			device
				.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
				.expect("create hessian shader module")
		};

		let entry_name = c"main";
		let stage = vk::PipelineShaderStageCreateInfo::default()
			.stage(vk::ShaderStageFlags::COMPUTE)
			.module(shader_module)
			.name(entry_name);

		let pipeline = unsafe {
			device
				.create_compute_pipelines(
					vk::PipelineCache::null(),
					&[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)],
					None,
				)
				.expect("create hessian compute pipeline")[0]
		};

		unsafe { device.destroy_shader_module(shader_module, None) };

		(
			pipeline,
			pipeline_layout,
			set_layout,
			descriptor_pool,
			descriptor_set,
		)
	}

	// /// Dynamically inspects the host OS and Vulkan physical device to initialize the arena
	// /// with zero hardcoded constraints.
	// /// The `device` parameter may have been created from an instance that is already dropped;
	// /// we reload its function pointers via vkGetDeviceProcAddr so they remain valid.
	// pub unsafe fn initialize_controller_from_hardware(
	// 	instance: &ash::Instance,
	// 	physical_device: vk::PhysicalDevice,
	// 	device: ash::Device,
	// 	queue: vk::Queue,
	// 	allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
	// ) -> MemoryController {
	// 	// ── 0. Reload device function pointers independently of the instance ──
	// 	// The `device` was created by instance.create_device(), which loaded VFNs via
	// 	// vkGetInstanceProcAddr. If the Instance is dropped (e.g. init_gpu() returns),
	// 	// those tables may dangle. Reload with vkGetDeviceProcAddr instead.
	// 	let raw_device = device.handle();
	// 	let entry = unsafe { ash::Entry::load() }.expect("load Entry");
	// 	// Re-create a Device that owns its own VFN table (survives instance drop).
	// 	// vkGetInstanceProcAddr with a null instance is valid per the Vulkan spec
	// 	// and returns device-local function pointers.


	// 	eprintln!("[CONTROLLER] initialize_controller_from_hardware START");

	// 	// ── 1. Query OS for Available System Memory (CPU) ──
	// 	let mut sys = System::new_all();
	// 	sys.refresh_memory();
	// 	let cpu_bytes = sys.available_memory();
	// 	eprintln!("[CONTROLLER] CPU available: {} bytes", cpu_bytes);

	// 	// ── 2. Query Vulkan Device for Device-Local Memory (VRAM) ──
	// 	let mem_properties =
	// 		unsafe { instance.get_physical_device_memory_properties(physical_device) };
	// 	let mut vram_bytes = 0u64;
	// 	for i in 0..mem_properties.memory_heap_count as usize {
	// 		let heap = mem_properties.memory_heaps[i];
	// 		if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
	// 			vram_bytes = vram_bytes.max(heap.size);
	// 		}
	// 	}
	// 	eprintln!("[CONTROLLER] VRAM: {} bytes", vram_bytes);

	// 	// ── 3. Resolve queue family for command pool ──
	// 	let queue_family_props =
	// 		unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
	// 	let queue_family = queue_family_props
	// 		.iter()
	// 		.position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
	// 		.unwrap_or_else(|| {
	// 			queue_family_props
	// 				.iter()
	// 				.position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
	// 				.expect("No suitable queue family")
	// 		}) as u32;

	// 	// ── 4. Create persistent command pool ──
	// 	let command_pool = unsafe {
	// 		device
	// 			.create_command_pool(
	// 				&vk::CommandPoolCreateInfo::default()
	// 					.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
	// 					.queue_family_index(queue_family),
	// 				None,
	// 			)
	// 			.expect("Failed to create command pool")
	// 	};

	// 	// ── 5. Calculate Arena Layout Constraints ──
	// 	let reserved = 4_000_000_000u64;
	// 	let total_addressable = (cpu_bytes + vram_bytes).saturating_sub(reserved);
	// 	let page_size: vk::DeviceSize = 64 * 1024; // 64 KiB pages
	// 	let total_pages = (total_addressable / page_size) as usize;

	// 	// ── 5a. Create the sparse buffer arena first (pipeline needs its handle) ──
	// 	let arena = unsafe {
	// 		VirtualTensorArena::new(&device, allocator.clone(), total_addressable, page_size)
	// 	};

	// 	// ── 5b. Load and compile quantize shader (binds descriptor set to sparse buffer) ──
	// 	eprintln!("[CONTROLLER] Creating quantize pipeline...");
	// 	let (quantize_pipeline, pipeline_layout, set_layout, pool, descriptor_set) =
	// 		Self::create_quantize_pipeline(&device, arena.sparse_buffer, page_size, total_pages);
	// 	eprintln!("[CONTROLLER] Pipeline created OK");

	// 	// ── 6. Instantiate Structural Ecosystem ──
	// 	let gpu = GpuContext::new(
	// 		device.clone(),
	// 		physical_device,
	// 		queue,
	// 		queue_family,
	// 		allocator,
	// 		command_pool,
	// 		quantize_pipeline,
	// 		pipeline_layout,
	// 		set_layout,
	// 		pool,
	// 		descriptor_set,
	// 	);
	// 	let cpu = CpuMemoryManager::new();

	// 	MemoryController {
	// 		arena,
	// 		gpu,
	// 		cpu,
	// 		max_cpu_bytes: cpu_bytes,
	// 		used_cpu_bytes: 0,
	// 		max_vram_bytes: vram_bytes,
	// 		used_vram_bytes: 0,
	// 	}
	// }



/// The `device` parameter may have been created from an instance that is already dropped;
/// we reload its function pointers via vkGetDeviceProcAddr so they remain valid.
pub unsafe fn initialize_controller_from_hardware(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
) -> MemoryController {
    eprintln!("[CONTROLLER] initialize_controller_from_hardware START");

    // ── 1. Query OS for Available System Memory (CPU) ──
    let mut sys = System::new_all();
    sys.refresh_memory();
    let cpu_bytes = sys.available_memory();
    eprintln!("[CONTROLLER] CPU available: {} bytes", cpu_bytes);

    // ── 2. Query Vulkan Device for Device-Local Memory (VRAM) ──
    // Fully safe now because `instance` is guaranteed to be alive
    let mem_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let mut vram_bytes = 0u64;
    for i in 0..mem_properties.memory_heap_count as usize {
        let heap = mem_properties.memory_heaps[i];
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            vram_bytes = vram_bytes.max(heap.size);
        }
    }
    eprintln!("[CONTROLLER] VRAM: {} bytes", vram_bytes);

    // ── 3. Resolve queue family for command pool ──
    let queue_family_props =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let queue_family = queue_family_props
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .unwrap_or_else(|| {
            queue_family_props
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .expect("No suitable queue family")
        }) as u32;

    // ── 4. Create persistent command pool ──
    let command_pool = unsafe {
        device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_family),
                None,
            )
            .expect("Failed to create command pool")
    };

    // ── 5. Calculate Arena Layout Constraints ──
    let reserved = 4_000_000_000u64;
    let total_addressable = (cpu_bytes + vram_bytes).saturating_sub(reserved);
    let page_size: vk::DeviceSize = 64 * 1024; // 64 KiB pages
    let total_pages = (total_addressable / page_size) as usize;

    // ── 5a. Create the sparse buffer arena first ──
    let arena = unsafe {
        VirtualTensorArena::new(&device, allocator.clone(), total_addressable, page_size)
    };

    // ── 5b. Load and compile quantize shader ──
    eprintln!("[CONTROLLER] Creating quantize pipeline...");
    let (quantize_pipeline, pipeline_layout, set_layout, pool, descriptor_set) =
        Self::create_quantize_pipeline(&device, arena.sparse_buffer, page_size, total_pages);
    eprintln!("[CONTROLLER] Pipeline created OK");

    // ── 6. Instantiate Structural Ecosystem ──
    let gpu = GpuContext::new(
        device,
				physical_device,
        queue,
				queue_family,
				allocator,
        command_pool,
        quantize_pipeline,
        pipeline_layout,
        set_layout,
        pool,
        descriptor_set,
    );
    let cpu = CpuMemoryManager::new();

    MemoryController {
        arena,
        gpu,
        cpu,
        max_cpu_bytes: cpu_bytes,
        used_cpu_bytes: 0,
        max_vram_bytes: vram_bytes,
        used_vram_bytes: 0,
    }
}


	/// Page a batch of model blocks into the arena, back to back.
	///
	/// `src` is the whole source model (typically an mmap of the file); each
	/// block names a byte range within it.
	///
	/// Blocks are concatenated into one contiguous byte stream before being cut
	/// into pages, so a page may straddle a block boundary and only the final
	/// page of the whole model is zero-padded. This is load-bearing: the arena
	/// offset of block *n* must equal the sum of the sizes of blocks 0..n, or
	/// tensors no longer sit where the sequential layout says they do.
	///
	/// Returns the number of bytes paged in.
	pub fn submit_blocks_for_paging(
		&mut self,
		src: &[u8],
		blocks: &[BlockDescriptor],
	) -> Result<u64, String> {
		use rayon::prelude::*;

		let page_size = self.arena.page_size as usize;
		let total_pages = self.arena.total_pages;

		let total_bytes: u64 = blocks.iter().map(|b| b.size).sum();
		let pages_needed = (total_bytes as usize).div_ceil(page_size);
		if pages_needed > total_pages {
			return Err(format!(
				"Model needs {} pages ({} bytes) but the arena only has {}",
				pages_needed, total_bytes, total_pages
			));
		}

		let tasks = plan_pages(src, blocks, page_size)?;

		// Commit before writing. `write_page` only uploads to the sparse buffer for
		// pages that are already GpuResident; an Unmapped page is routed to CPU RAM
		// instead, which leaves the shader reading unbacked memory as zeros.
		let page_indices: Vec<usize> = tasks.iter().map(|(idx, _)| *idx).collect();
		self.commit_pages(&page_indices);

		// Partition tasks into GPU (batched) and CPU (individual) groups.
		let (gpu_pages, cpu_pages): (Vec<_>, Vec<_>) = tasks.into_iter().partition(|(idx, _)| {
			self.arena.page_table[*idx].residency == PageResidency::GpuResident
		});

		// Batch ALL GPU page uploads into ONE staging buffer + ONE submit.
		// Build regions with sequential staging offsets so each page copies from
		// its own slice of the staging buffer, not all from offset 0.
		if !gpu_pages.is_empty() {
			let page_size = self.arena.page_size;
			let mut stg_offset: vk::DeviceSize = 0;
			let regions: Vec<(vk::DeviceSize, vk::DeviceSize, vk::DeviceSize)> = gpu_pages
				.iter()
				.map(|(idx, data)| {
					let dst_offset = *idx as vk::DeviceSize * page_size;
					let size = data.len() as vk::DeviceSize;
					let cur_stg = stg_offset;
					stg_offset += size;
					(cur_stg, dst_offset, size)
				})
				.collect();
			let data_slices: Vec<&[u8]> = gpu_pages.iter().map(|(_, d)| d.as_slice()).collect();
			unsafe {
				self.gpu
					.batch_upload(self.arena.sparse_buffer, &regions, &data_slices);
			}
		}

		// CPU-bound pages written serially (CpuMemoryManager is not Send/Sync).
		// The GPU batch is the performance-critical path — CPU writes are fast.
		let page_size_usize = self.arena.page_size as usize;
		for (page_idx, data) in cpu_pages {
			self.cpu.write_page(page_idx, page_size_usize, &data);
			self.arena.page_table[page_idx].residency = PageResidency::CpuResident;
			self.arena.page_table[page_idx].cpu_offset = Some(page_idx * page_size_usize);
		}

		Ok(total_bytes)
	}


	/// Clone just enough state for parallel page writes.
	pub fn clone_for_parallel(&self) -> MemoryController {
		Self {
			arena: self.arena.clone(),
			gpu: self.gpu.clone_shallow(),
			cpu: CpuMemoryManager::new(),
			max_cpu_bytes: self.max_cpu_bytes,
			used_cpu_bytes: 0, // each parallel worker tracks its own usage
			max_vram_bytes: self.max_vram_bytes,
			used_vram_bytes: 0,
		}
	}
}

/// Copy one block out of the source model, bounds-checked.
///
/// Errors rather than returning zeros: a buffer of zeros is indistinguishable from
/// real weights downstream and quantizes into a plausible-looking file.
fn read_model_block(src: &[u8], offset: u64, size: u64) -> Result<&[u8], String> {
	let start = offset as usize;
	let end = start
		.checked_add(size as usize)
		.ok_or_else(|| format!("Block offset {} + size {} overflows", offset, size))?;
	src.get(start..end).ok_or_else(|| {
		format!(
			"Block [{}, {}) is outside the {}-byte source model",
			start,
			end,
			src.len()
		)
	})
}

/// Cut the concatenation of `blocks` into page-sized writes.
///
/// The blocks are treated as one contiguous stream, so a page may straddle a block
/// boundary and only the last page is zero-padded. Keeping interior boundaries
/// unpadded is what makes an arena offset equal a model offset.
fn plan_pages(
	src: &[u8],
	blocks: &[BlockDescriptor],
	page_size: usize,
) -> Result<Vec<(usize, Vec<u8>)>, String> {
	let mut tasks: Vec<(usize, Vec<u8>)> = Vec::new();
	let mut next_page = 0usize;
	let mut carry: Vec<u8> = Vec::with_capacity(page_size);

	for block in blocks {
		let bytes = read_model_block(src, block.offset, block.size)?;

		let mut pos = 0usize;
		while pos < bytes.len() {
			let need = page_size - carry.len();
			let take = need.min(bytes.len() - pos);
			carry.extend_from_slice(&bytes[pos..pos + take]);
			pos += take;

			if carry.len() == page_size {
				tasks.push((next_page, std::mem::take(&mut carry)));
				carry.reserve(page_size);
				next_page += 1;
			}
		}
	}

	// Only the tail of the model is padded — never an interior boundary.
	if !carry.is_empty() {
		carry.resize(page_size, 0);
		tasks.push((next_page, carry));
	}

	Ok(tasks)
}

/// Rayon's `for_each_with` needs an owned handle per worker. Cloning shares the
/// arena and the Vulkan handles while giving each worker its own CPU accounting.
impl Clone for MemoryController {
	fn clone(&self) -> Self {
		self.clone_for_parallel()
	}
}

/// GPU buffer layout matching TensorArenaArchitecture.pdf spec.
///
/// Memory layout (contiguous, in order):
///   1. ModelSize          — 4 bytes  (u32: total block count)
///   2. GPUWorkPool        — blocks × 4 bytes  (u32 per block, bits 30/31 are status)
///   3. BlockSizeBuffer    — blocks × 4 bytes  (u32: high16=width, low16=height)
///   4. BlockData          — blocks × block_size × 4 bytes  (f32 weights)
///   5. Buckets            — blocks × bucket_region_size
///
/// Per-block bucket region: 100 bucket entries.
/// Each bucket: u8 prefix_index + u16[BlockSize/100] tail_indices.

/// Initialize the global MemoryController from Vulkan hardware.
/// Must be called once before any gpu_quantize() calls.
/// `entry` keeps libvulkan loaded; if it drops the device VFN table dangles.
pub fn init_global_controller(
	entry: ash::Entry,
	instance: &ash::Instance,
	physical_device: ash::vk::PhysicalDevice,
	device: ash::Device,
	queue: ash::vk::Queue,
	allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
) -> Result<(), String> {
	// Pin the Entry so libvulkan.so never unloads while this process is alive.
	GLOBAL_ENTRY.set(entry).map_err(|_| "Entry already initialized".to_string())?;

	let ctrl = unsafe {
		MemoryController::initialize_controller_from_hardware(
			instance,
			physical_device,
			device,
			queue,
			allocator,
		)
	};
	GLOBAL_CONTROLLER
		.set(Arc::new(Mutex::new(ctrl)))
		.map_err(|_| "Global controller already initialized".to_string())
}

/// Global Vulkan Entry — keeps libvulkan loaded while the process is alive.
/// Dropping this would unload the library and invalidate every device VFN.
pub static GLOBAL_ENTRY: OnceLock<ash::Entry> = OnceLock::new();

/// Global MemoryController — initialized once by init_global_controller().
pub static GLOBAL_CONTROLLER: OnceLock<Arc<Mutex<MemoryController>>> = OnceLock::new();

#[cfg(test)]
mod tests {
	use super::*;

	/// Flatten a page plan back into the arena's byte image.
	fn arena_image(tasks: &[(usize, Vec<u8>)], page_size: usize) -> Vec<u8> {
		let mut out = vec![0u8; tasks.len() * page_size];
		for (page_idx, data) in tasks {
			let start = page_idx * page_size;
			out[start..start + data.len()].copy_from_slice(data);
		}
		out
	}

	/// The invariant the whole format rests on: a block's arena offset is the sum
	/// of the sizes of the blocks before it. No interior padding, no reordering.
	#[test]
	fn blocks_land_at_their_running_sum_offset() {
		let page_size = 64;
		// Sizes deliberately not multiples of the page size.
		let sizes: [u64; 4] = [100, 30, 77, 5];

		// Each block gets its own byte value so misplacement is visible.
		let mut src = Vec::new();
		let mut blocks = Vec::new();
		for (i, &size) in sizes.iter().enumerate() {
			blocks.push(BlockDescriptor {
				offset: src.len() as u64,
				size,
			});
			src.extend(std::iter::repeat_n(b'A' + i as u8, size as usize));
		}

		let tasks = plan_pages(&src, &blocks, page_size).expect("plan");
		let image = arena_image(&tasks, page_size);

		let mut expected_offset = 0usize;
		for (i, &size) in sizes.iter().enumerate() {
			let tag = b'A' + i as u8;
			let got = &image[expected_offset..expected_offset + size as usize];
			assert!(
				got.iter().all(|&b| b == tag),
				"block {} is not contiguous at offset {}",
				i,
				expected_offset
			);
			expected_offset += size as usize;
		}

		// The arena image is byte-identical to the concatenated source.
		assert_eq!(&image[..src.len()], &src[..]);
	}

	#[test]
	fn only_the_final_page_is_padded() {
		let page_size = 64;
		let src = vec![0xABu8; 200];
		let blocks = [BlockDescriptor {
			offset: 0,
			size: 200,
		}];

		let tasks = plan_pages(&src, &blocks, page_size).expect("plan");
		assert_eq!(tasks.len(), 200usize.div_ceil(page_size));

		let image = arena_image(&tasks, page_size);
		assert!(image[..200].iter().all(|&b| b == 0xAB));
		// 200 = 3 pages + 8 bytes; the tail of the last page is the only padding.
		assert!(image[200..].iter().all(|&b| b == 0));
	}

	#[test]
	fn pages_are_numbered_consecutively_from_zero() {
		let page_size = 16;
		let src = vec![1u8; 100];
		let blocks = [
			BlockDescriptor {
				offset: 0,
				size: 40,
			},
			BlockDescriptor {
				offset: 40,
				size: 60,
			},
		];

		let tasks = plan_pages(&src, &blocks, page_size).expect("plan");
		for (i, (page_idx, data)) in tasks.iter().enumerate() {
			assert_eq!(*page_idx, i, "page indices must be dense and ordered");
			assert_eq!(data.len(), page_size, "every write is exactly one page");
		}
	}

	#[test]
	fn out_of_range_block_is_an_error_not_zeros() {
		let src = vec![0u8; 10];
		let blocks = [BlockDescriptor {
			offset: 8,
			size: 99,
		}];
		assert!(plan_pages(&src, &blocks, 64).is_err());
	}

	// ── Task 1: dual-binding layout sizing invariants ────────────────────────

	#[test]
	fn ping_pong_activation_matches_spec_bytes() {
		// 5,120 hidden dim at 4,096 tokens → exactly 83,886,080 bytes (80.00 MiB).
		let act = PingPongActivation::new(4096, 5120);
		assert_eq!(act.block_bytes, 83_886_080);

		// The two blocks are identical and tile the activation arena with no gap.
		let (r_off, r_len) = act.read_input_range();
		let (s_off, s_len) = act.write_scratch_range();
		assert_eq!(r_len, s_len);
		assert_eq!(r_off + r_len, s_off, "read block and scratch block must be adjacent");
		assert_eq!(s_off + s_len, 2 * act.block_bytes);

		// Advancing flips roles: what was scratch is now the read input.
		let mut a = PingPongActivation::new(4096, 5120);
		let before = a.read_input_range().0;
		a.advance();
		assert_eq!(a.read_input_range().0, s_off);
		assert_ne!(before, a.read_input_range().0);
	}

	#[test]
	fn square_attention_tiles_exactly() {
		// 5120 × 5120 FP16 → exactly 52,428,800 bytes (50.00 MiB), a 160×160 tile grid.
		let page = WeightTilePage { n_dim_in: 5120, n_dim_out: 5120, offset_bytes: 0 };
		assert_eq!(page.size_bytes(), 52_428_800);
		assert_eq!(page.tile_grid(), (160, 160));
		assert_eq!(page.tile_count(), 25_600);
	}

	#[test]
	fn swiglu_ffn_expansion_tiles_exactly() {
		// The 5,120-hidden baseline checkpoint's real FFN intermediate is 17,408 (a ×3.4
		// expansion — NOT 8/3; older notes mislabeled it). It comes from the model config,
		// so we pass it in and only assert it tiles by 32 and sizes correctly.
		let inter = LayerBindingLayout::swiglu_intermediate(17_408);
		assert_eq!(inter, 17_408); // already a multiple of 32 → unchanged

		let page = WeightTilePage { n_dim_in: 5120, n_dim_out: inter, offset_bytes: 0 };
		assert_eq!(page.size_bytes(), 178_257_920);
		assert_eq!(page.tile_grid(), (160, 544));
		assert_eq!(page.tile_count(), 87_040);

		// A non-tileable intermediate gets rounded up to the next 32-multiple.
		assert_eq!(LayerBindingLayout::swiglu_intermediate(17_409), 17_440);
	}

	#[test]
	fn weight_pages_pack_by_running_sum() {
		// Square attention + FFN up + FFN gate packed back-to-back: each page's offset is
		// the sum of every earlier page's size, and the total matches the arena bytes.
		let shapes = [(5120u64, 5120u64), (5120, 17408), (5120, 17408)];
		let space = WeightTileSpace::new(&shapes);
		let pages = space.plan_pages(&shapes);

		assert_eq!(pages.len(), shapes.len());
		let mut expected_offset = 0u64;
		for p in &pages {
			assert_eq!(p.offset_bytes, expected_offset, "page offset must be the running sum");
			expected_offset += p.size_bytes();
		}
		assert_eq!(expected_offset, space.total_bytes);
		assert_eq!(space.tile_bytes, WeightTilePage::TILE_BYTES);
		assert_eq!(WeightTilePage::TILE_BYTES, 2048); // 32×32 FP16 = one 2 KB tile
	}
}

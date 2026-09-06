use crate::memory_controller::virtual_tensor_arena::{
	OperationType, PageResidency, VirtualTensorArena,
};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use std::sync::{Arc, Mutex, OnceLock};
use sysinfo::System;
use std::ffi::c_void;
use std::ffi::CStr;

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
			device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty());
		}
		pooled.lock().unwrap().push(cmd);
	}

	/// Synchronous upload: copy `data` into `buf` at `offset` via a staging buffer.
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
		eprintln!(
			"[UPLOAD] t+{:3}ms  creating staging buffer...",
			t.elapsed().as_millis()
		);
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

		// 2. Allocate host-visible mapped memory for the staging buffer
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
		eprintln!(
			"[UPLOAD] t+{:3}ms  writing {} bytes to staging memory",
			t.elapsed().as_millis(),
			data.len()
		);
		if let Some(ptr) = alloc.mapped_ptr() {
			unsafe {
				std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>().as_ptr(), data.len());
			}
		} else {
			panic!("Staging allocation is not host-mapped");
		}

		// 4. Record copy command
		eprintln!(
			"[UPLOAD] t+{:3}ms  allocating command buffer...",
			t.elapsed().as_millis()
		);
		use core::mem;
		let cp_raw = unsafe { mem::transmute::<vk::CommandPool, u64>(self.command_pool) };
		let q_raw = unsafe { mem::transmute::<vk::Queue, u64>(self.queue_handle) };
		eprintln!("[UPLOAD]    command_pool = 0x{:016X}", cp_raw);
		eprintln!("[UPLOAD]    queue        = 0x{:016X}", q_raw);

		unsafe {
			self.device_handle
				.queue_wait_idle(self.queue_handle)
				.expect("queue_wait_idle before cmd alloc");
		}
		eprintln!(
			"[UPLOAD] t+{:3}ms  queue idle, getting command buffer...",
			t.elapsed().as_millis()
		);

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
				buf,
				&[vk::BufferCopy::default()
					.src_offset(0)
					.dst_offset(offset)
					.size(size)],
			);
			self.device_handle
				.end_command_buffer(cmd)
				.expect("end command buffer");
		}

		// 5. Submit and wait
		eprintln!(
			"[UPLOAD] t+{:3}ms  creating fence + submitting...",
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
				.expect("submit upload");
			eprintln!(
				"[UPLOAD] t+{:3}ms  queue_submit done, waiting on fence (this may block)...",
				t.elapsed().as_millis()
			);
			self.device_handle
				.wait_for_fences(&[fence], true, u64::MAX)
				.expect("wait upload fence");
			eprintln!(
				"[UPLOAD] t+{:3}ms  fence signaled, upload complete",
				t.elapsed().as_millis()
			);
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
		for (page_idx, _) in &tasks {
			self.commit_page(*page_idx);
		}

		// Serial, deliberately. A GpuResident page upload allocates a command
		// buffer from `gpu.command_pool` and submits to `gpu.queue` — both are
		// externally-synchronized Vulkan objects, so driving them from several
		// rayon workers at once is undefined behaviour, not a speedup. The earlier
		// parallel version survived only while models were small enough to fit a
		// single page; at 32+ pages it took the process down mid-upload.
		//
		// The uploads all funnel through one queue anyway, so there was no real
		// concurrency to win here. Parallelism belongs in the encode path, where
		// `quantize_cpu` already uses it.
		for (page_idx, data) in tasks {
			self.write_page(page_idx, &data);
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
}

use crate::memory_controller::{
	cpu_mem_op::CpuMemory,
	virtual_tensor_arena::{OperationType, PageResidency, VirtualTensorArena},
};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use std::sync::{Arc, Mutex};
use sysinfo::System;

/// A block of model data to be paged into the arena.
#[derive(Clone, Debug)]
pub struct BlockDescriptor {
	/// File/offset within the source model
	pub offset: u64,
	/// Number of bytes in this block
	pub size: u64,
}

// GPU context — holds Vulkan device, queue, allocator, and command pool handles.
pub struct GpuContext {
	pub device_handle: ash::Device,
	pub physical_device: vk::PhysicalDevice,
	pub queue_handle: vk::Queue,
	pub queue_family: u32,
	pub allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
	pub command_pool: vk::CommandPool,
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
		eprintln!("[UPLOAD] t+{:3}ms  queue idle, allocating command buffer...", t.elapsed().as_millis());

		let alloc_info = vk::CommandBufferAllocateInfo::default()
			.command_pool(self.command_pool)
			.level(vk::CommandBufferLevel::PRIMARY)
			.command_buffer_count(1);
		eprintln!("[UPLOAD]    alloc_info built, calling allocate_command_buffers...");
		let cmd_result = unsafe {
			self.device_handle.allocate_command_buffers(&alloc_info)
		};
		let desc = match &cmd_result { Ok(v) => format!("{} buffers", v.len()), Err(_) => "ERR".to_string() };
		eprintln!("[UPLOAD]    allocate_command_buffers returned: {}", desc);
		let cmd = match cmd_result {
			Ok(cmds) => cmds[0],
			Err(e) => panic!("[UPLOAD] allocate_command_buffers failed: {:?}", e),
		};
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
		let fence = unsafe {
			self.device_handle
				.create_fence(&vk::FenceCreateInfo::default(), None)
				.expect("create fence")
		};
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

		// 6. Cleanup
		unsafe {
			self.device_handle.destroy_fence(fence, None);
			self.device_handle
				.free_command_buffers(self.command_pool, &[cmd]);
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

		// 3. Record copy command
		let cmd = unsafe {
			self.device_handle
				.allocate_command_buffers(
					&vk::CommandBufferAllocateInfo::default()
						.command_pool(self.command_pool)
						.level(vk::CommandBufferLevel::PRIMARY)
						.command_buffer_count(1),
				)
				.expect("allocate command buffer")[0]
		};
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

		// 4. Submit and wait
		eprintln!(
			"[DOWNLOAD] t+{:3}ms  creating fence + submitting...",
			t.elapsed().as_millis()
		);
		let fence = unsafe {
			self.device_handle
				.create_fence(&vk::FenceCreateInfo::default(), None)
				.expect("create fence")
		};
		unsafe {
			self.device_handle
				.queue_submit(
					self.queue_handle,
					&[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
					fence,
				)
				.expect("submit download");
			eprintln!("[DOWNLOAD] t+{:3}ms  queue_submit done, waiting on fence (BLOCKS HERE = GPU HANG)...", t.elapsed().as_millis());
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

		// 6. Cleanup
		unsafe {
			self.device_handle.destroy_fence(fence, None);
			self.device_handle
				.free_command_buffers(self.command_pool, &[cmd]);
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

	// src/memory_controller/controller.rs

    /// Dynamically inspects the host OS and Vulkan physical device to initialize the arena
    /// with zero hardcoded constraints.
    /// The `device` parameter may have been created from an instance that is already dropped;
    /// we reload its function pointers via vkGetDeviceProcAddr so they remain valid.
    pub unsafe fn initialize_controller_from_hardware(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        queue: vk::Queue,
        allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    ) -> MemoryController {
        // ── 0. Reload device function pointers independently of the instance ──
        // The `device` was created by instance.create_device(), which loaded VFNs via
        // vkGetInstanceProcAddr. If the Instance is dropped (e.g. init_gpu() returns),
        // those tables may dangle. Reload with vkGetDeviceProcAddr instead.
        use std::ffi::CStr;
        let raw_device = device.handle();
        let entry = unsafe { ash::Entry::load() }.expect("load Entry");
        eprintln!("[CONTROLLER] initialize_controller_from_hardware START");
        
        // ── 1. Query OS for Available System Memory (CPU) ──
        let mut sys = System::new_all();
        sys.refresh_memory();
        let cpu_bytes = sys.available_memory();
        eprintln!("[CONTROLLER] CPU available: {} bytes", cpu_bytes);
        
        // ── 2. Query Vulkan Device for Device-Local Memory (VRAM) ──
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
        
        // ── 5a. Create the sparse buffer arena first (pipeline needs its handle) ──
        let arena = unsafe {
            VirtualTensorArena::new(&device, allocator.clone(), total_addressable, page_size)
        };
        
        // ── 5b. Load and compile quantize shader (binds descriptor set to sparse buffer) ──
        eprintln!("[CONTROLLER] Creating quantize pipeline...");
        let (quantize_pipeline, pipeline_layout, set_layout, pool, descriptor_set) =
            Self::create_quantize_pipeline(&device, arena.sparse_buffer, page_size, total_pages);
        eprintln!("[CONTROLLER] Pipeline created OK");
        
        // ── 6. Instantiate Structural Ecosystem ──
        let gpu = GpuContext::new(
            device.clone(),
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


	/// Create the quantize compute pipeline: shader module → descriptor set layout →
	/// pipeline layout → pipeline → descriptor pool + set binding the sparse buffer.
	unsafe fn create_quantize_pipeline(
		device: &ash::Device,
		sparse_buffer: vk::Buffer,
		page_size: vk::DeviceSize,
		_total_pages: usize,
	) -> (
		vk::Pipeline,
		vk::PipelineLayout,
		vk::DescriptorSetLayout,
		vk::DescriptorPool,
		vk::DescriptorSet,
	) {
		use std::include_bytes;

		// ── Load SPIR-V module ──
		eprintln!("[PIPELINE] Loading SPIR-V bytes...");
		let spirv_bytes: &[u8] = include_bytes!("../models/dedupe/quantize_gemv.spv");
		eprintln!("[PIPELINE] SPIR-V loaded: {} bytes", spirv_bytes.len());

		let spirv_words: Vec<u32> = spirv_bytes
			.chunks_exact(4)
			.map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
			.collect();
		eprintln!("[PIPELINE] SPIR-V words: {}", spirv_words.len());

		eprintln!("[PIPELINE] Creating shader module...");
		let shader_module = device
			.create_shader_module(
				&vk::ShaderModuleCreateInfo::default().code(&spirv_words),
				None,
			)
			.expect("create_shader_module");
		eprintln!("[PIPELINE] Shader module created OK");

		// ── Descriptor set layout: 4 bindings ──
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
			vk::DescriptorSetLayoutBinding::default()
				.binding(3)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
		];
		eprintln!("[PIPELINE] Creating descriptor set layout with 4 bindings...");
		let set_layout = device
			.create_descriptor_set_layout(
				&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
				None,
			)
			.expect("create_descriptor_set_layout");
		eprintln!("[PIPELINE] Descriptor set layout created OK");

		// ── Pipeline layout: push constants (COMPUTE_SHADER, offset 0, 8 bytes = 2×u32) ──
		eprintln!("[PIPELINE] Creating pipeline layout...");
		let push_range = [vk::PushConstantRange::default()
			.stage_flags(vk::ShaderStageFlags::COMPUTE)
			.offset(0)
			.size(8)]; 
		let set_layouts = [set_layout];
		let pipeline_layout = device
			.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(&set_layouts)
					.push_constant_ranges(&push_range),
				None,
			)
			.expect("create_pipeline_layout");
		eprintln!("[PIPELINE] Pipeline layout created OK");

		// ── Pipeline ──
		eprintln!("[PIPELINE] Creating compute pipeline...");
		let entry = std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap();
		let stage = vk::PipelineShaderStageCreateInfo::default()
			.stage(vk::ShaderStageFlags::COMPUTE)
			.module(shader_module)
			.name(entry);
		let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
			.stage(stage)
			.layout(pipeline_layout);
		let pipeline = device
			.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
			.expect("create_compute_pipelines")[0];
		eprintln!("[PIPELINE] Compute pipeline created OK");

		// ── Descriptor pool: 4 bindings/set × 64 sets = 256 descriptors ──
		eprintln!("[PIPELINE] Creating descriptor pool (256 slots)...");
		let pool_sizes = [vk::DescriptorPoolSize {
			ty: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 256,
		}];
		let pool = device
			.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo::default()
					.max_sets(64)
					.flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
					.pool_sizes(&pool_sizes),
				None,
			)
			.expect("create_descriptor_pool");
		eprintln!("[PIPELINE] Descriptor pool created OK");

		// ── Allocate descriptor set ──
		eprintln!("[PIPELINE] Allocating descriptor set...");
		let descriptor_set = device
			.allocate_descriptor_sets(
				&vk::DescriptorSetAllocateInfo::default()
					.descriptor_pool(pool)
					.set_layouts(&[set_layout]),
			)
			.expect("allocate_descriptor_sets")[0];
		eprintln!("[PIPELINE] Descriptor set allocated OK");

		// ── HERE IS THE FIX: Calculate offsets inside your single sparse buffer ──
		// Adjust these multiplier values depending on how many pages each section needs!
		let binding0_offset = 0;
		let binding0_range  = page_size * 100; // Example: SourceWeights takes 100 pages

		let binding1_offset = binding0_offset + binding0_range;
		let binding1_range  = page_size * 50;  // Example: GpuWorkPool takes 50 pages

		let binding2_offset = binding1_offset + binding1_range;
		let binding2_range  = page_size * 10;  // Example: GlobalCounters takes 10 pages

		let binding3_offset = binding2_offset + binding2_range;
		let binding3_range  = vk::WHOLE_SIZE;  // Remainder goes to BucketData

		// Define specific descriptor buffers details per binding slot
		let info_binding0 = vk::DescriptorBufferInfo::default()
			.buffer(sparse_buffer).offset(binding0_offset).range(binding0_range);

		let info_binding1 = vk::DescriptorBufferInfo::default()
			.buffer(sparse_buffer).offset(binding1_offset).range(binding1_range);

		let info_binding2 = vk::DescriptorBufferInfo::default()
			.buffer(sparse_buffer).offset(binding2_offset).range(binding2_range);

		let info_binding3 = vk::DescriptorBufferInfo::default()
			.buffer(sparse_buffer).offset(binding3_offset).range(binding3_range);

		let desc_writers = [
			vk::WriteDescriptorSet::default()
				.dst_set(descriptor_set)
				.dst_binding(0)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.buffer_info(std::slice::from_ref(&info_binding0)),
			vk::WriteDescriptorSet::default()
				.dst_set(descriptor_set)
				.dst_binding(1)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.buffer_info(std::slice::from_ref(&info_binding1)),
			vk::WriteDescriptorSet::default()
				.dst_set(descriptor_set)
				.dst_binding(2)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.buffer_info(std::slice::from_ref(&info_binding2)),
			vk::WriteDescriptorSet::default()
				.dst_set(descriptor_set)
				.dst_binding(3)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.buffer_info(std::slice::from_ref(&info_binding3)),
		];
		
		unsafe { device.update_descriptor_sets(&desc_writers, &[]); }
		eprintln!("[PIPELINE] Descriptor set bound to sparse buffer subdivisions OK");

		device.destroy_shader_module(shader_module, None);
		(pipeline, pipeline_layout, set_layout, pool, descriptor_set)
	}

	// ── Block paging (work-stealing threadpool) ────────────────────────────

	/// Submit a batch of model blocks for paging into the arena.
	///
	/// Each block is described by `(offset, size)` — offset within the source
	/// model file, size in bytes. Blocks are distributed across a rayon threadpool
	/// (work-stealing scheduler). Each block is written to its assigned page slot.
	///
	/// Page assignment: block `i` → page `i` (linear mapping).
	/// Blocks larger than one page are split across consecutive pages.
	pub fn submit_blocks_for_paging(&mut self, blocks: &[BlockDescriptor]) {
		use rayon::prelude::*;

		let page_size = self.arena.page_size as u64;
		let total_pages = self.arena.total_pages;

		// Build a flat list of (page_index, data) tasks.
		// Blocks that span multiple pages are split.
		let mut tasks: Vec<(usize, Vec<u8>)> = Vec::new();
		let mut next_page = 0usize;

		for block in blocks {
			let block_size = block.size as usize;
			let block_offset = block.offset as usize;
			let bytes = self.read_model_block(block.offset, block.size);

			let mut pos = 0usize;
			while pos < bytes.len() && next_page < total_pages {
				let remaining = bytes.len() - pos;
				let chunk_size = remaining.min(page_size as usize);
				let end = pos + chunk_size;

				// Pad the last chunk of a block to page_size if needed
				let chunk = if end == bytes.len() && remaining < page_size as usize {
					let mut padded = vec![0u8; page_size as usize];
					padded[..remaining].copy_from_slice(&bytes[pos..]);
					padded
				} else {
					bytes[pos..end].to_vec()
				};

				tasks.push((next_page, chunk));
				next_page += 1;
				pos = end;
			}
		}

		// Parallel write via rayon work-stealing pool.
		// Each task writes to its page; the controller handles residency routing.
		let controller = Arc::new(Mutex::new(self.clone_for_parallel()));
		tasks.into_par_iter().for_each(|(page_idx, data)| {
			let mut ctrl = controller.lock().unwrap();
			ctrl.write_page(page_idx, &data);
		});
	}

	/// Read a block from the source model. Placeholder — wire to real loader.
	fn read_model_block(&self, _offset: u64, _size: u64) -> Vec<u8> {
		// TODO: mmap the model file and read the block
		vec![0u8; _size as usize]
	}

	/// Clone just enough state for parallel page writes.
	fn clone_for_parallel(&self) -> MemoryController {
		Self {
			arena: self.arena.clone_shallow(),
			gpu: self.gpu.clone_shallow(),
			cpu: CpuMemoryManager::new(),
			max_cpu_bytes: self.max_cpu_bytes,
			used_cpu_bytes: 0, // each parallel worker tracks its own usage
			max_vram_bytes: self.max_vram_bytes,
			used_vram_bytes: 0,
		}
	}
}

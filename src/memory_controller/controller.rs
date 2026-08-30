use crate::memory_controller::virtual_tensor_arena::{
	OperationType, PageResidency, VirtualTensorArena,
};
use ash::vk;
use std::sync::{Arc, Mutex};
use sysinfo::System;

const CPU_MEMORY_AVAIL: u64 = sysinfo::System::available_memory();
/// A block of model data to be paged into the arena.
#[derive(Clone, Debug)]
pub struct BlockDescriptor {
	/// File/offset within the source model
	pub offset: u64,
	/// Number of bytes in this block
	pub size: u64,
}

// GPU context — holds Vulkan device, queue, and allocator handles.
pub struct GpuContext {
	pub device_handle: ash::Device,
	pub queue_handle: vk::Queue,
	pub allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
}

impl GpuContext {
	pub fn new(
		device: ash::Device,
		queue: vk::Queue,
		allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
	) -> Self {
		Self {
			device_handle: device,
			queue_handle: queue,
			allocator,
		}
	}
	/// Shallow clone for parallel worker contexts — shares Arc handles.
	pub fn clone_shallow(&self) -> Self {
		Self {
			device_handle: self.device_handle.clone(),
			queue_handle: self.queue_handle,
			allocator: Arc::clone(&self.allocator),
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
	pub unsafe fn upload(&self, buf: vk::Buffer, offset: vk::DeviceSize, data: &[u8]) {
		// TODO: implement via allocator mapped write or buffer copy
		let _ = (buf, offset, data);
	}
	pub unsafe fn download(
		&self,
		buf: vk::Buffer,
		offset: vk::DeviceSize,
		size: vk::DeviceSize,
	) -> Vec<u8> {
		// TODO: implement via buffer copy to host-visible staging buffer
		let _ = (buf, offset, size);
		vec![0u8; size as usize]
	}
}

/// CPU memory backing pool — wraps the cpu_mem_op pool.
pub struct CpuMemoryManager {
	pool: crate::memory_controller::cpu_mem_op::CpuMemory,
}

impl CpuMemoryManager {
	pub fn new(total_size: usize) -> Self {
		Self {
			pool: crate::memory_controller::cpu_mem_op::CpuMemory::new(total_size),
		}
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

	// ── Initialization ───────────────────────────────────────────────────

	/// Create a MemoryController backed by a VirtualTensorArena.
	///
	/// Total addressable memory = cpu_bytes + vram_bytes − 4 GB (reserved).
	/// Pages are 256 bytes each.
	pub unsafe fn init(
		gpu: GpuContext,
		cpu_bytes: u64,
		vram_bytes: u64,
	) -> Self {
		let reserved = 4_000_000_000u64; // 4 GB headroom
		let total = (cpu_bytes + vram_bytes).saturating_sub(reserved);
		let page_size: vk::DeviceSize = 256;
		let total_pages = (total / page_size) as usize;

		let allocator = gpu.allocator();
		let arena = VirtualTensorArena::new(
			gpu.device(),
			Arc::clone(&allocator),
			total,
			page_size,
		);

		let cpu_pool_size = (total_pages * page_size as usize)
			.min(cpu_bytes as usize)
			.max(1);
		let cpu = CpuMemoryManager::new(cpu_pool_size);

		Self {
			arena,
			gpu,
			cpu,
			max_cpu_bytes: cpu_bytes,
			used_cpu_bytes: 0,
			max_vram_bytes: vram_bytes,
			used_vram_bytes: 0,
		}
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
			let mut block_offset = block.offset as usize;
			let bytes = self.read_model_block(block.offset, block.size);

			let mut pos = 0usize;
			while pos < bytes.len() && next_page < total_pages {
				let remaining = bytes.len() - pos;
				let chunk_size = (block_size - remaining).min(page_size as usize);
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
		// Shallow clone — page writes don't need GPU allocations
		Self {
			arena: self.arena.clone_shallow(),
			gpu: self.gpu.clone_shallow(),
			cpu: CpuMemoryManager::new(self.cpu.pool.capacity()),
			max_cpu_bytes: self.max_cpu_bytes,
			used_cpu_bytes: self.used_cpu_bytes,
			max_vram_bytes: self.max_vram_bytes,
			used_vram_bytes: self.used_vram_bytes,
		}
	}
}

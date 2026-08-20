use ash::vk;
use gpu_allocator::vulkan::{Allocator, Allocation, AllocationCreateDesc};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PageResidency {
	Unmapped,   // Not committed to physical memory
	GpuResident,
	CpuResident
}

pub enum OperationType {
	Move,
	Drop
}

pub struct VirtualPage {
	pub residency: PageResidency,
	pub gpu_allocation: Option<Allocation>,
	pub cpu_offset: Option<usize>,
}

pub struct VirtualTensorArena {
	pub total_virtual_size: vk::DeviceSize,
	pub page_size: vk::DeviceSize,
	pub total_pages: usize,

	// Main large buffer w/ no memory bounds
	pub sparse_buffer: vk::Buffer,

	// Routing Table
	pub page_table: Vec<VirtualPage>,

	pub allocator: Arc<Mutex<Allocator>>,

}

impl VirtualTensorArena {
	pub unsafe fn new(
			device: &ash::Device,
			allocator: Arc<Mutex<Allocator>>,
			total_virtual_size: vk::DeviceSize,
			page_size: vk::DeviceSize,
	) -> Self {
			let total_pages = (total_virtual_size / page_size) as usize;

			// --- GPU Virtual Reservation ---
			// Create a buffer flagged with sparse binding and sparse residency
			// Allowing buffer to exist in virtual address space w/o physical memory backing
			let buffer_create_info = vk::BufferCreateInfo::default()
					.size(total_virtual_size)
					.usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC )
					.sharing_mode(vk::SharingMode::EXCLUSIVE)
					.flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);

			let buffer = unsafe { device.create_buffer(&buffer_create_info, None)};

			let sparse_buffer = buffer.expect("Failed to create sparse buffer virtual shell.");

			let mut page_table = Vec::with_capacity(total_pages);

			for _ in 0..total_pages {
					page_table.push(VirtualPage {
							residency: PageResidency::Unmapped,
							gpu_allocation: None,
							cpu_offset: None,
					});
			}

			Self {
					total_virtual_size,
					page_size,
					total_pages,
					sparse_buffer,
					page_table,
					allocator
			}

	}

	pub unsafe fn commit_page (
			&mut self,
			device: &ash::Device,
			bind_queue: vk::Queue,
			page_index: usize,
	) {

			let allocator_clone = Arc::clone(&self.allocator);

			let page = &mut self.page_table[page_index];

			if page.residency != PageResidency::Unmapped {
					return; // Has been mapped previously
			}

			let offset = page_index as vk::DeviceSize * self.page_size;

			// Query memory requirements from sparse buffer segment
			let mem_reqs: vk::MemoryRequirements = unsafe { device.get_buffer_memory_requirements(self.sparse_buffer)};

			let mut allocator_guard = allocator_clone.lock().unwrap();

			let gpu_alloc_result = allocator_guard.allocate(&AllocationCreateDesc {
					name: "tensor_page_gpu",
					requirements: vk::MemoryRequirements {
							size: self.page_size,
							alignment: mem_reqs.alignment,
							memory_type_bits: mem_reqs.memory_type_bits,
					},
					location: gpu_allocator::MemoryLocation::GpuOnly,
					linear: true,
					allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
			});

			drop(allocator_guard);

			match gpu_alloc_result {
					Ok(allocation) => {

							let allocation_offset = allocation.offset();

							let raw_memory_handle = unsafe { allocation.memory() };


									let memory_bind = vk::SparseMemoryBind::default()
											.resource_offset(offset)
											.size(self.page_size)
											.memory(raw_memory_handle)
											.memory_offset(allocation_offset);

									let buffer_bind_info = vk::SparseBufferMemoryBindInfo::default()
											.buffer(self.sparse_buffer)
											.binds(std::slice::from_ref(&memory_bind));

									let bind_info = vk::BindSparseInfo::default()
											.buffer_binds(std::slice::from_ref(&buffer_bind_info));

									let bind_result = unsafe {
											device.queue_bind_sparse(bind_queue, &[bind_info], vk::Fence::null())
									};

									match bind_result {
											Ok(_) => {
													page.residency = PageResidency::GpuResident;
													page.gpu_allocation = Some(allocation);
													println!("Page {} successfully mapped to GPU physical memory.", page_index);
											}
											Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {

													// FIX: Drop explicit reference inside block, then safely route
													let mut allocator_guard = allocator_clone.lock().unwrap();
													let _ = allocator_guard.free(allocation);

													self.route_page_to_cpu(page_index);

											}
											Err(e) => panic!("Unrecoverable Vulkan sparse bind error: {:?}", e),
									}
							}

					// CATCH HOST / ALLOCATOR OOM
					Err(_) => {
							self.route_page_to_cpu(page_index);
					}
			}
	}

	fn route_page_to_cpu(&mut self, page_index: usize) {
		let page = &mut self.page_table[page_index];
		page.residency = PageResidency::CpuResident;
		page.cpu_offset = Some(page_index * self.page_size as usize);
		println!("GPU OOM Detected! Page {} intelligently routed to CPU memory fallback.", page_index);
	}

	pub unsafe fn evict_page(
		&mut self,
		page_index: usize,
		allocator: Arc<Mutex<Allocator>>,
		bind_queue: vk::Queue,
		device: &ash::Device
	) {

		let offset = page_index as vk::DeviceSize * self.page_size;
		let memory_bind = vk::SparseMemoryBind::default()
			.resource_offset(offset)
			.size(self.page_size)
			.memory(vk::DeviceMemory::null())
			.memory_offset(0);
		let buffer_bind_info = vk::SparseBufferMemoryBindInfo::default()
			.buffer(self.sparse_buffer)
			.binds(std::slice::from_ref(&memory_bind));
		let bind_info = vk::BindSparseInfo::default()
			.buffer_binds(std::slice::from_ref(&buffer_bind_info));

		let bind_result = unsafe {
			device.queue_bind_sparse(bind_queue, &[bind_info], vk::Fence::null())
		};

		match bind_result {
			Ok(_) => {
				if let Some(allocation) = self.page_table[page_index].gpu_allocation.take() {
					let mut allocator_guard = allocator.lock().unwrap();
					let _ = allocator_guard.free(allocation);
					let page = &mut self.page_table[page_index];
					page.residency = PageResidency::Unmapped;
					page.cpu_offset = None;
				}
			},
			Err(vk::Result::ERROR_DEVICE_LOST) => {
				// GPU is gone - Everything is dead, nothing else matters
				panic!("Device lost during sparse unbind of page {}", page_index)
			},
			Err(e) => {
				panic!("Unrecoverable sparse unbind error for page {}: {:?}", page_index, e);
			}
		}
	}
}

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PageResidency {
	Unmapped,
	GpuResident,
	CpuResident,
}

pub enum OperationType {
	Move,
	Drop,
}

pub struct VirtualPage {
	pub residency: PageResidency,
	pub gpu_allocation: Option<Allocation>,
	pub cpu_offset: Option<usize>,
}

impl Clone for VirtualPage {
	fn clone(&self) -> Self {
		Self {
			residency: self.residency,
			gpu_allocation: None, // Allocation isn't Clone; workers don't need it for page writes
			cpu_offset: self.cpu_offset,
		}
	}
}

pub struct VirtualTensorArena {
	pub total_virtual_size: vk::DeviceSize,
	pub page_size: vk::DeviceSize,
	pub total_pages: usize,
	pub sparse_buffer: vk::Buffer,
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

		let buffer_create_info = vk::BufferCreateInfo::default()
			.size(total_virtual_size)
			.usage(
				vk::BufferUsageFlags::STORAGE_BUFFER
					| vk::BufferUsageFlags::TRANSFER_DST
					| vk::BufferUsageFlags::TRANSFER_SRC,
			)
			.sharing_mode(vk::SharingMode::EXCLUSIVE)
			.flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);

		let buffer = unsafe { device.create_buffer(&buffer_create_info, None) };
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
			allocator,
		}
	}

	pub unsafe fn commit_page(
		&mut self,
		device: &ash::Device,
		bind_queue: vk::Queue,
		page_index: usize,
	) {
		let allocator_clone = Arc::clone(&self.allocator);
		let page = &mut self.page_table[page_index];

		if page.residency != PageResidency::Unmapped {
			return;
		}

		let offset = page_index as vk::DeviceSize * self.page_size;
		let mem_reqs: vk::MemoryRequirements =
			unsafe { device.get_buffer_memory_requirements(self.sparse_buffer) };
		let mut allocator_guard = allocator_clone.lock().unwrap();

		// 1. Build the allocation footprint descriptor variable explicitly
		let alloc_desc = AllocationCreateDesc {
			name: "tensor_page_gpu",
			requirements: vk::MemoryRequirements {
				size: self.page_size,
				alignment: mem_reqs.alignment,
				memory_type_bits: mem_reqs.memory_type_bits,
			},
			location: gpu_allocator::MemoryLocation::GpuOnly,
			linear: true,
			allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
		};

		// 2. Pass a BORROWED REFERENCE (&alloc_desc) to the true allocate function
		let gpu_alloc_result = allocator_guard.allocate(&alloc_desc);

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

								let fence = unsafe {
					device.create_fence(&vk::FenceCreateInfo::default(), None)
						.expect("create fence for sparse bind")
				};

				let bind_result =
					unsafe { device.queue_bind_sparse(bind_queue, &[bind_info], fence) };
				bind_result.expect("Queue bind sparse submission failed");

				// --- VALID VULKAN QUEUE PUMP ---
				// Submitting an empty SubmitInfo array to the queue family forces the 
				// asynchronous sparse timeline to immediately flush and execute.
				unsafe {
					device.queue_submit(bind_queue, &[], vk::Fence::null())
						.expect("Queue flush submission failed");
				}
				// -------------------------------

				// Your Exact Debugging Instrumentation Layout
				let t_bind = std::time::Instant::now();
				eprintln!("[VTA] page={} waiting on sparse bind fence...", page_index);
				unsafe {
					device
						.wait_for_fences(&[fence], true, u64::MAX)
						.expect("wait sparse bind fence");
					device.destroy_fence(fence, None);
				}
				eprintln!(
					"[VTA] page={} sparse bind fence signaled after {:?}",
					page_index,
					t_bind.elapsed()
				);



				match bind_result {
					Ok(_) => {
						page.residency = PageResidency::GpuResident;
						page.gpu_allocation = Some(allocation);
						println!(
							"Page {} successfully mapped to GPU physical memory.",
							page_index
						);
					}
					Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
						let mut allocator_guard = allocator_clone.lock().unwrap();
						let _ = allocator_guard.free(allocation);
						self.route_page_to_cpu(page_index);
					}
					Err(e) => panic!("Unrecoverable Vulkan sparse bind error: {:?}", e),
				}
			}
			Err(_) => {
				self.route_page_to_cpu(page_index);
			}
		}
	}

	fn route_page_to_cpu(&mut self, page_index: usize) {
		let page = &mut self.page_table[page_index];
		page.residency = PageResidency::CpuResident;
		page.cpu_offset = Some(page_index * self.page_size as usize);
		println!(
			"GPU OOM Detected! Page {} intelligently routed to CPU memory fallback.",
			page_index
		);
	}

	pub unsafe fn evict_page(
		&mut self,
		page_index: usize,
		allocator: Arc<Mutex<Allocator>>,
		bind_queue: vk::Queue,
		device: &ash::Device,
		op_type: OperationType,
	) {
		let offset = page_index as vk::DeviceSize * self.page_size;
		let page = &mut self.page_table[page_index];
		if page.residency != PageResidency::GpuResident {
			return;
		}
		if let Some(allocation) = page.gpu_allocation.take() {
			let raw_memory = unsafe { allocation.memory() };
			let memory_bind = vk::SparseMemoryBind::default()
				.resource_offset(offset)
				.size(self.page_size)
				.memory(raw_memory)
				.memory_offset(allocation.offset());
			let buffer_bind_info = vk::SparseBufferMemoryBindInfo::default()
				.buffer(self.sparse_buffer)
				.binds(std::slice::from_ref(&memory_bind));
			let bind_info =
				vk::BindSparseInfo::default().buffer_binds(std::slice::from_ref(&buffer_bind_info));

			// Create a real fence to force synchronization
			let fence = unsafe {
				device
					.create_fence(&vk::FenceCreateInfo::default(), None)
					.expect("create fence for sparse unbind")
			};

			let _ = unsafe { device.queue_bind_sparse(bind_queue, &[bind_info], fence) };

			// Block the CPU thread until the unbind is fully completed by the GPU
			unsafe {
				device
					.wait_for_fences(&[fence], true, u64::MAX)
					.expect("wait sparse unbind fence");
				device.destroy_fence(fence, None);
			}

			let mut allocator_guard = allocator.lock().unwrap();
			let _ = allocator_guard.free(allocation);
		}
		page.residency = PageResidency::Unmapped;
		page.gpu_allocation = None;
		println!("Page {} safely evicted from GPU.", page_index);
	}

	pub fn clone_shallow(&self) -> Self {
		Self {
			total_virtual_size: self.total_virtual_size,
			page_size: self.page_size,
			total_pages: self.total_pages,
			sparse_buffer: self.sparse_buffer,
			page_table: self.page_table.clone(),
			allocator: Arc::clone(&self.allocator),
		}
	}
}

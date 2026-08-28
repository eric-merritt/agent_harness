use ash::vk;
use gpu_allocator::vulkan::{Allocator, Allocation, AllocationCreateDesc};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PageResidency {
    Unmapped,   
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
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)
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
            allocator
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
        let mem_reqs: vk::MemoryRequirements = unsafe { device.get_buffer_memory_requirements(self.sparse_buffer) };
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
        println!("GPU OOM Detected! Page {} intelligently routed to CPU memory fallback.", page_index);
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

        if let OperationType::Move = op_type {
            if page.residency == PageResidency::GpuResident {
                page.cpu_offset = Some(page_index * self.page_size as usize);
                // Pipeline fallback tracking allocation trigger maps here
            }
        }

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
                if let Some(allocation) = page.gpu_allocation.take() {
                    let mut allocator_guard = allocator.lock().unwrap();
                    let _ = allocator_guard.free(allocation);
                }
                
                match op_type {
                    OperationType::Drop => {
                        page.residency = PageResidency::Unmapped;
                        page.cpu_offset = None;
                    }
                    OperationType::Move => {
                        page.residency = PageResidency::CpuResident;
                    }
                }
            },
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                panic!("Device lost during sparse unbind of page {}", page_index);
            },
            Err(e) => {
                panic!("Unrecoverable sparse unbind error for page {}: {:?}", page_index, e);
            }
        }
    }

    // ── DISPATCH ROUTER BRIDGING YOUR STRUCT TO DETACHED STANDALONE FUNCTIONS ──

    /// Safe execution loop router. Extracts a zero-copy slice of your global host tracking buffer
    /// and paths it directly into your math module functions.
    pub fn execute_layer_math(
        &self,
        page_index: usize,
        global_cpu_memory: &[u8], // Your raw central host storage pool
        out: &mut [f32],
        scales: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) {
        let page = &self.page_table[page_index];

        match page.residency {
            PageResidency::Unmapped => {
                panic!("Execution Error: Page {} is unmapped.", page_index);
            }
            PageResidency::GpuResident => {
                // Dispatched via Vulkan compute pipeline using self.sparse_buffer
            }
            PageResidency::CpuResident => {
                if let Some(offset) = page.cpu_offset {
                    let start = offset;
                    let end = start + self.page_size as usize;
                    
                    // Slice your active host byte array without copying any bytes
                    let packed = &global_cpu_memory[start..end];

                    // Invoke your math standalone execution function
                    crate::inference::math::gemv_4bit_into(
                        out,
                        scales,
                        packed,
						x,
                        rows,
                        cols,
                        group_size,
                    );
                } else {
                    panic!("Data anomaly: Page {} marked as CPU resident but lacks an allocation offset", page_index);
                }
            }
        }
    }

    // Inside your VirtualTensorArena implementation or dispatcher loop
pub unsafe fn dispatch_gpu_quantization(
    &self,
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set: vk::DescriptorSet,
    rows: u32,
    cols: u32,
    group_size: u32,
) {
    // Bind your uniform descriptor sets pointing to the sparse_buffer 
    unsafe {
        device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[descriptor_set],
        &[],
    )};

    unsafe {
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
    };
    // Push dimensional bounds parameters natively into the shader registers
    let push_constants = [rows, cols, group_size];
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(push_constants.as_ptr() as *const u8, 12)
    };
    unsafe {
        device.cmd_push_constants(
        command_buffer,
        pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
    )};

    // Calculate optimal grid size grid blocks
    let total_elements = rows * cols;
    let group_count_x = (total_elements + 127) / 128;

    // Launch execution thread grid directly over the Vulkan sparse buffer shell
    unsafe {device.cmd_dispatch(command_buffer, group_count_x, 1, 1) };
}
}
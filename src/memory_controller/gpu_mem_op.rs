use ash::vk;
use ash::{Entry, Instance};
use gpu_allocator::vulkan::{Allocator, Allocation, AllocationCreateDesc, AllocatorCreateDesc};
use std::sync::{Arc, Mutex};

#[allow(dead_code)]
pub struct GpuMemory {
    entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    command_pool: vk::CommandPool,
    allocator: Arc<Mutex<Allocator>>,
    vram_capacity: u64,
}

/// Handle to an in-flight async transfer.
/// Must be consumed via `wait_transfer` (uploads) or `finish_download` (downloads).
/// If dropped unconsumed, Drop waits for the fence and cleans up —
/// but the handle must not outlive the GpuMemory that created it.
pub struct TransferHandle {
    inner: Option<TransferInner>,
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
}

struct TransferInner {
    fence: vk::Fence,
    staging_buffer: vk::Buffer,
    staging_alloc: Allocation,
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
}

impl TransferHandle {
    /// Non-blocking check: has the transfer completed?
    pub fn is_done(&self) -> bool {
        let Some(ref inner) = self.inner else { return true; };
        unsafe { self.device.get_fence_status(inner.fence).unwrap_or(false) }
    }
}

impl Drop for TransferHandle {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            unsafe {
                let _ = self.device.wait_for_fences(&[inner.fence], true, u64::MAX);
                self.device.destroy_fence(inner.fence, None);
                self.device.free_command_buffers(inner.command_pool, &[inner.command_buffer]);
                self.device.destroy_buffer(inner.staging_buffer, None);
            }
            let mut guard = self.allocator.lock().unwrap();
            let _ = guard.free(inner.staging_alloc);
        }
    }
}

impl GpuMemory {
    pub fn new() -> Self {
        let entry = Entry::linked();
        let instance = unsafe {
            entry.create_instance(&vk::InstanceCreateInfo::default(), None)
                .expect("Failed to create Vulkan instance")
        };

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .expect("Failed to enumerate physical devices");
        if physical_devices.is_empty() {
            panic!("No Vulkan physical devices found");
        }

        let (physical_device, vram_capacity) = physical_devices.iter()
            .map(|pd| {
                let props = unsafe {
                    instance.get_physical_device_memory_properties(*pd)
                };
                let vram: u64 = (0..props.memory_heap_count as usize)
                    .filter(|&i| props.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                    .map(|i| props.memory_heaps[i].size)
                    .sum();
                (*pd, vram)
            })
            .max_by_key(|(_, vram)| *vram)
            .expect("No physical devices with memory");

        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
        log::info!(
            "GpuMemory: {} — {:.1} GB VRAM",
            device_props.device_name_as_c_str().unwrap_or_default().to_string_lossy(),
            vram_capacity as f64 / 1e9
        );

        let queue_family_properties = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };
        let queue_family = queue_family_properties.iter()
            .position(|props| {
                props.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && props.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING)
            })
            .unwrap_or_else(|| {
                queue_family_properties.iter()
                    .position(|props| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                    .expect("No suitable queue family found")
            });

        let queue_priorities = [1.0f32];
        let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family as u32)
            .queue_priorities(&queue_priorities)];

        let device = unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_create_infos),
                None,
            ).expect("Failed to create logical device")
        };
        let queue = unsafe { device.get_device_queue(queue_family as u32, 0) };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        }).expect("Failed to create GPU allocator");
        let allocator = Arc::new(Mutex::new(allocator));

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_family as u32),
                None,
            ).expect("Failed to create command pool")
        };

        Self {
            entry, instance, physical_device, device, queue,
            queue_family: queue_family as u32, command_pool, allocator, vram_capacity,
        }
    }

    pub fn device(&self) -> &ash::Device { &self.device }
    pub fn queue(&self) -> vk::Queue { self.queue }
    pub fn allocator(&self) -> Arc<Mutex<Allocator>> { Arc::clone(&self.allocator) }
    pub fn vram_capacity(&self) -> u64 { self.vram_capacity }

    // ── Synchronous transfers (thin wrappers over async) ─────────────

    pub unsafe fn upload(&self, dest_buffer: vk::Buffer, dest_offset: vk::DeviceSize, data: &[u8]) {
        let handle = unsafe { self.upload_async(dest_buffer, dest_offset, data) };
        self.wait_transfer(handle);
    }

    pub unsafe fn download(&self, src_buffer: vk::Buffer, src_offset: vk::DeviceSize, size: vk::DeviceSize) -> Vec<u8> {
        let handle = unsafe { self.download_async(src_buffer, src_offset, size) };
        self.finish_download(handle)
    }

    // ── Async transfers (non-blocking) ─────────────────────────────────

    /// Start an async upload. Data copied to staging immediately, GPU copy
    /// submitted with a fence. Call `wait_transfer` before using the dest data.
    pub unsafe fn upload_async(
        &self,
        dest_buffer: vk::Buffer,
        dest_offset: vk::DeviceSize,
        data: &[u8],
    ) -> TransferHandle {
        let size = data.len() as vk::DeviceSize;

        let staging_buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            ).expect("Failed to create staging buffer")
        };
        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(staging_buffer) };

        let mut guard = self.allocator.lock().unwrap();
        let staging_alloc = guard.allocate(&AllocationCreateDesc {
            name: "staging_upload",
            requirements: mem_reqs,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        }).expect("Failed to allocate staging memory");
        drop(guard);

        unsafe {
            self.device.bind_buffer_memory(staging_buffer, staging_alloc.memory(), staging_alloc.offset())
                .expect("Failed to bind staging buffer memory");
        }

        if let Some(ptr) = staging_alloc.mapped_ptr() {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast().as_ptr(), data.len()); }
        } else {
            panic!("Staging allocation is not host-mapped");
        }

        let cmd = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            ).expect("Failed to allocate command buffer")[0]
        };

        unsafe {
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .expect("Failed to begin command buffer");
            self.device.cmd_copy_buffer(cmd, staging_buffer, dest_buffer,
                &[vk::BufferCopy::default().src_offset(0).dst_offset(dest_offset).size(size)]);
            self.device.end_command_buffer(cmd).expect("Failed to end command buffer");
        }

        let fence = unsafe {
            self.device.create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("Failed to create fence")
        };

        unsafe {
            self.device.queue_submit(self.queue,
                &[vk::SubmitInfo::default()
                    .command_buffers(std::slice::from_ref(&cmd))],
                fence,
            ).expect("Failed to submit async upload");
        }

        TransferHandle {
            inner: Some(TransferInner {
                fence, staging_buffer, staging_alloc, command_buffer: cmd,
                command_pool: self.command_pool,
            }),
            device: self.device.clone(),
            allocator: Arc::clone(&self.allocator),
        }
    }

    /// Start an async download. GPU copy submitted with a fence.
    /// Call `finish_download` to wait and retrieve the data.
    pub unsafe fn download_async(
        &self,
        src_buffer: vk::Buffer,
        src_offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> TransferHandle {
        if size == 0 {
            return TransferHandle {
                inner: None,
                device: self.device.clone(),
                allocator: Arc::clone(&self.allocator),
            };
        }

        let staging_buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            ).expect("Failed to create staging buffer")
        };
        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(staging_buffer) };

        let mut guard = self.allocator.lock().unwrap();
        let staging_alloc = guard.allocate(&AllocationCreateDesc {
            name: "staging_download",
            requirements: mem_reqs,
            location: gpu_allocator::MemoryLocation::GpuToCpu,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        }).expect("Failed to allocate staging memory");
        drop(guard);

        unsafe {
            self.device.bind_buffer_memory(staging_buffer, staging_alloc.memory(), staging_alloc.offset())
                .expect("Failed to bind staging buffer memory");
        }

        let cmd = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            ).expect("Failed to allocate command buffer")[0]
        };

        unsafe {
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .expect("Failed to begin command buffer");
            self.device.cmd_copy_buffer(cmd, src_buffer, staging_buffer,
                &[vk::BufferCopy::default().src_offset(src_offset).dst_offset(0).size(size)]);
            self.device.end_command_buffer(cmd).expect("Failed to end command buffer");
        }

        let fence = unsafe {
            self.device.create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("Failed to create fence")
        };

        unsafe {
            self.device.queue_submit(self.queue,
                &[vk::SubmitInfo::default()
                    .command_buffers(std::slice::from_ref(&cmd))],
                fence,
            ).expect("Failed to submit async download");
        }

        TransferHandle {
            inner: Some(TransferInner {
                fence, staging_buffer, staging_alloc, command_buffer: cmd,
                command_pool: self.command_pool,
            }),
            device: self.device.clone(),
            allocator: Arc::clone(&self.allocator),
        }
    }

    // ── Transfer completion ─────────────────────────────────────────────

    /// Block until a transfer completes, then clean up. Consumes the handle.
    pub fn wait_transfer(&self, mut handle: TransferHandle) {
        let Some(inner) = handle.inner.take() else { return; };
        unsafe {
            self.device.wait_for_fences(&[inner.fence], true, u64::MAX)
                .expect("Failed to wait for transfer fence");
            self.device.destroy_fence(inner.fence, None);
            self.device.free_command_buffers(inner.command_pool, &[inner.command_buffer]);
            self.device.destroy_buffer(inner.staging_buffer, None);
        }
        let mut guard = self.allocator.lock().unwrap();
        let _ = guard.free(inner.staging_alloc);
    }

    /// Block until a download completes, read staging, then clean up. Consumes the handle.
    pub fn finish_download(&self, mut handle: TransferHandle) -> Vec<u8> {
        let Some(inner) = handle.inner.take() else { return Vec::new(); };
        unsafe {
            self.device.wait_for_fences(&[inner.fence], true, u64::MAX)
                .expect("Failed to wait for download fence");
        }

        let size = inner.staging_alloc.size() as usize;
        let mut result = vec![0u8; size];
        if let Some(ptr) = inner.staging_alloc.mapped_ptr() {
            unsafe { std::ptr::copy_nonoverlapping(ptr.cast().as_ptr(), result.as_mut_ptr(), size); }
        }

        unsafe {
            self.device.destroy_fence(inner.fence, None);
            self.device.free_command_buffers(inner.command_pool, &[inner.command_buffer]);
            self.device.destroy_buffer(inner.staging_buffer, None);
        }
        let mut guard = self.allocator.lock().unwrap();
        let _ = guard.free(inner.staging_alloc);

        result
    }
}

impl Drop for GpuMemory {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

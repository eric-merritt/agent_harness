// gpu_mem_ops.rs 

/// Initialize Vulkan for GPU benchmarks.
/// Returns a tuple containing the Entry loader (to keep the library from being unmapped) 
/// along with the plain Instance, PhysicalDevice, Device, Queue, and Allocator.
pub fn init_gpu() -> Result<(
    ash::Entry, 
    ash::Instance, 
    ash::vk::PhysicalDevice, 
    ash::Device, 
    ash::vk::Queue, 
    std::sync::Arc<std::sync::Mutex<gpu_allocator::vulkan::Allocator>>
), String> {
	use ash::vk;
	use gpu_allocator::vulkan::AllocatorCreateDesc;
	use gpu_allocator::AllocationSizes;

	eprintln!("[GPU INIT] Starting Vulkan initialization...");

	// ── Create Vulkan instance ──
	let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("Vulkan entry load failed: {:?}", e))?;
	eprintln!("[GPU INIT] Vulkan entry loaded OK");

	let app_name = c"compress_bench";
	let app_info = vk::ApplicationInfo::default()
		.application_name(&app_name)
		.api_version(vk::API_VERSION_1_3);
	let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
	let instance = unsafe { entry.create_instance(&instance_create_info, None) }
		.map_err(|e| format!("create_instance failed: {:?}", e))?;
	eprintln!("[GPU INIT] Instance created OK");

	// ── Pick first physical device ──
	let phys_devices = unsafe { instance.enumerate_physical_devices() }
		.map_err(|e| format!("enumerate_physical_devices failed: {:?}", e))?;
	if phys_devices.is_empty() {
		return Err("No Vulkan physical devices found".into());
	}
	let physical_device = phys_devices[0];
	eprintln!("[GPU INIT] Found {} physical device(s), using first", phys_devices.len());

	// ── Get a compute-capable queue ──
	let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
	let queue_family_index = queue_families
		.iter()
		.position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
		.ok_or("No compute queue family found")? as u32;
	eprintln!("[GPU INIT] Compute queue family index: {}", queue_family_index);

	let queue_family = vk::DeviceQueueCreateInfo::default()
		.queue_family_index(queue_family_index)
		.queue_priorities(&[1.0]);

	let enabled_features = vk::PhysicalDeviceFeatures::default()
		.sparse_binding(true)
		.sparse_residency_buffer(true);
	let queue_family_list = [queue_family];
	let device_create_info = vk::DeviceCreateInfo::default()
		.queue_create_infos(&queue_family_list)
		.enabled_features(&enabled_features);

	let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
		.map_err(|e| format!("create device failed: {:?}", e))?;
	eprintln!("[GPU INIT] Device created OK");

	let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
	eprintln!("[GPU INIT] Queue acquired OK");

	// This now receives the standard plain instance clone as expected by gpu_allocator!
	let allocator = gpu_allocator::vulkan::Allocator::new(&AllocatorCreateDesc {
		instance: instance.clone(), 
		device: device.clone(),
		physical_device,
		debug_settings: Default::default(),
		buffer_device_address: false,
		allocation_sizes: AllocationSizes::default(),
	})
	.map_err(|e| format!("allocator create failed: {:?}", e))?;
	eprintln!("[GPU INIT] Allocator created OK");

	// Return the entry as item 0 to keep the libvulkan module active in memory
	Ok((entry, instance, physical_device, device, queue, std::sync::Arc::new(std::sync::Mutex::new(allocator))))
}

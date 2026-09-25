// Temporary probe: query Vulkan sparse residency properties of the discrete GPU.
#![allow(unused)]
use ash::vk;

fn main() -> std::process::ExitCode { run() }

fn run() -> std::process::ExitCode {
	let entry = match unsafe { ash::Entry::load() } {
		Ok(e) => e,
		Err(e) => {
			eprintln!("Vulkan entry load failed: {:?}", e);
			return std::process::ExitCode::FAILURE;
		}
	};
	let app_name = c"sparse_probe";
	let app_info = vk::ApplicationInfo::default()
		.application_name(&app_name)
		.api_version(vk::API_VERSION_1_3);
	let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
	let instance = match unsafe { entry.create_instance(&instance_create_info, None) } {
		Ok(i) => i,
		Err(e) => {
			eprintln!("create_instance failed: {:?}", e);
			return std::process::ExitCode::FAILURE;
		}
	};

	let phys_devices = match unsafe { instance.enumerate_physical_devices() } {
		Ok(d) => d,
		Err(e) => {
			eprintln!("enumerate_physical_devices failed: {:?}", e);
			return std::process::ExitCode::FAILURE;
		}
	};
	if phys_devices.is_empty() {
		eprintln!("No Vulkan physical devices found");
		return std::process::ExitCode::FAILURE;
	}
	// Prefer a discrete GPU (the 4080 Super), fall back to device 0.
	let physical_device = phys_devices
		.iter()
		.copied()
		.find(|&d| {
			unsafe { instance.get_physical_device_properties(d) }.device_type
				== vk::PhysicalDeviceType::DISCRETE_GPU
		})
		.unwrap_or(phys_devices[0]);

	let props = unsafe { instance.get_physical_device_properties(physical_device) };
	let name = props
		.device_name_as_c_str()
		.map(|s| s.to_string_lossy().into_owned())
		.unwrap_or_default();
	println!("Device: {}", name);
	println!(
		"API version: {}.{}.{}\n",
		vk::api_version_major(props.api_version),
		vk::api_version_minor(props.api_version),
		vk::api_version_patch(props.api_version)
	);

	let sparse = props.sparse_properties;
	println!("PhysicalDeviceSparseProperties:");
	println!(
		"  residencyStandard2DBlockShape: {}",
		sparse.residency_standard2_d_block_shape as u32
	);
	println!(
		"  residencyStandard2DMultisampleBlockShape: {}",
		sparse.residency_standard2_d_multisample_block_shape as u32
	);
	println!(
		"  residencyStandard3DBlockShape: {}",
		sparse.residency_standard3_d_block_shape as u32
	);
	println!(
		"  residencyAlignedMipSize: {}",
		sparse.residency_aligned_mip_size as u32
	);
	println!(
		"  residencyNonResidentStrict: {}",
		sparse.residency_non_resident_strict as u32
	);

	let features = unsafe { instance.get_physical_device_features(physical_device) };
	println!();
	println!("Features relevant to sparse residency:");
	println!("  sparseBinding: {}", features.sparse_binding as u32);
	println!(
		"  sparseResidencyBuffer: {}",
		features.sparse_residency_buffer as u32
	);
	println!(
		"  sparseResidencyImage2D: {}",
		features.sparse_residency_image2_d as u32
	);
	println!(
		"  sparseResidencyImage3D: {}",
		features.sparse_residency_image3_d as u32
	);
	println!(
		"  sparseResidencyAliased: {}",
		features.sparse_residency_aliased as u32
	);

	let limits = props.limits;
	println!("\nLimits relevant to sparse residency:");
	println!(
		"  sparseAddressSpaceSize: {} bytes ({} GB)",
		limits.sparse_address_space_size,
		limits.sparse_address_space_size / (1024 * 1024 * 1024)
	);

	// Query per-format sparse residency granularity for the formats we actually use.
	// This tells us the minimum tile size the driver will map — i.e. our page size.
	println!("\nSparse image format properties (granularity = min mappable tile):");
	for (fmt_name, fmt) in [
		("R8", vk::Format::R8_UNORM),
		("R16", vk::Format::R16_UNORM),
		("R32", vk::Format::R32_SFLOAT),
		("RGBA8", vk::Format::R8G8B8A8_UNORM),
	] {
		let props = unsafe {
			instance.get_physical_device_sparse_image_format_properties(
				physical_device,
				fmt,
				vk::ImageType::TYPE_1D,
				vk::SampleCountFlags::TYPE_1,
				vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
				vk::ImageTiling::OPTIMAL,
			)
		};
		for p in props {
			let g = p.image_granularity;
			let flags = format!("{:?}", p.flags);
			println!(
				"  {} 1D: granularity={}x{}x{} {}",
				fmt_name, g.width, g.height, g.depth, flags
			);
		}
	}

	std::process::ExitCode::SUCCESS
}

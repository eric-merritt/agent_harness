// benches/decompress_bench.rs
//
// Decompression benchmarks with error measurement.
// Compression is done once upfront; only decompression is timed.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::sync::{Arc, Mutex};

use agent_harness::models::dedupe::tensor::DedupCountTensor;
use agent_harness::models::dedupe::compressor::init_global_controller;

/// Deterministic pseudo-random weights (LCG → Box-Muller → N(0, σ²)).
/// σ=0.3 approximates typical transformer weight magnitudes.
fn make_weights(n: usize) -> Vec<f32> {
	let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
	let mut out = Vec::with_capacity(n);
	let sigma = 0.3_f64;
	loop {
		state = state
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		let u1 = (((state >> 40) as f64) / (1u64 << 24) as f64)
			.max(1e-15)
			.min(1.0 - f64::EPSILON);
		state = state
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		let u2 = ((state >> 40) as f64) / (1u64 << 24) as f64;
		let r = (-2.0_f64 * u1.ln()).sqrt();
		let theta = u2 * 2.0 * std::f64::consts::PI;
		out.push((r * theta.cos() * sigma) as f32);
		if out.len() >= n {
			break;
		}
		out.push((r * theta.sin() * sigma) as f32);
		if out.len() >= n {
			break;
		}
	}
	out.truncate(n);
	out
}

/// Initialize Vulkan + global MemoryController for GPU benchmarks.
fn init_gpu() -> Result<(), String> {
	use ash::vk;
	use gpu_allocator::vulkan::AllocatorCreateDesc;
	use gpu_allocator::AllocationSizes;

	let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("Vulkan entry load failed: {:?}", e))?;
	let app_name = c"decompress_bench";
	let app_info = vk::ApplicationInfo::default()
		.application_name(&app_name)
		.api_version(vk::API_VERSION_1_2);
	let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
	let instance = unsafe { entry.create_instance(&instance_create_info, None) }
		.map_err(|e| format!("create_instance failed: {:?}", e))?;

	let phys_devices = unsafe { instance.enumerate_physical_devices() }
		.map_err(|e| format!("enumerate_physical_devices failed: {:?}", e))?;
	if phys_devices.is_empty() {
		return Err("No Vulkan physical devices found".into());
	}
	let physical_device = phys_devices[0];

	let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
	let queue_family_index = queue_families
		.iter()
		.position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
		.ok_or("No compute queue family found")? as u32;

	let queue_family = vk::DeviceQueueCreateInfo::default()
		.queue_family_index(queue_family_index)
		.queue_priorities(&[1.0]);

	let enabled_features = vk::PhysicalDeviceFeatures::default();
	let queue_family_list = [queue_family];
	let device_create_info = vk::DeviceCreateInfo::default()
		.queue_create_infos(&queue_family_list)
		.enabled_features(&enabled_features);

	let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
		.map_err(|e| format!("create device failed: {:?}", e))?;

	let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

	let allocator = gpu_allocator::vulkan::Allocator::new(&AllocatorCreateDesc {
		instance,
		device: device.clone(),
		physical_device,
		debug_settings: Default::default(),
		buffer_device_address: false,
		allocation_sizes: AllocationSizes::default(),
	})
	.map_err(|e| format!("allocator create failed: {:?}", e))?;

	// Dummy instance for controller init (needs &Instance for memory property query)
	let entry2 = unsafe { ash::Entry::load() }.map_err(|e| format!("Vulkan entry load failed: {:?}", e))?;
	let dummy_app = c"init_helper";
	let dummy_app_info = vk::ApplicationInfo::default()
		.application_name(&dummy_app)
		.api_version(vk::API_VERSION_1_2);
	let dummy_instance = unsafe { entry2.create_instance(&vk::InstanceCreateInfo::default().application_info(&dummy_app_info), None) }
		.map_err(|e| format!("dummy instance failed: {:?}", e))?;

	unsafe {
		init_global_controller(
			&dummy_instance,
			physical_device,
			device,
			queue,
			Arc::new(Mutex::new(allocator)),
		)
	}
}

/// Mean squared error between two vectors (same length).
fn mse(a: &[f32], b: &[f32]) -> f32 {
	let len = a.len().min(b.len());
	if len == 0 {
		return 0.0;
	}
	let sum: f32 = a
		.iter()
		.zip(b.iter())
		.map(|(&x, &y)| {
			let d = x - y;
			d * d
		})
		.sum();
	sum / len as f32
}

fn bench_decompress(c: &mut Criterion) {
	let n = 1_000_000;
	let prefix_digits = 2;
	let truncate_rounds = 3;
	let weights = make_weights(n);

	// ── Initialize GPU ──
	let gpu_init = init_gpu();
	if let Err(ref e) = gpu_init {
		eprintln!("decompress_bench: GPU init failed: {} — GPU benches will be skipped", e);
	}

	// Pre-compress all methods once
	let (tensor_sp, sandbag_sp) =
		DedupCountTensor::compress_quantized(&weights, prefix_digits, truncate_rounds);
	let (tensor_sk, sandbag_sk) =
		DedupCountTensor::compress_quantized_kl(&weights, prefix_digits, truncate_rounds);
	let (tensor_ap, sandbag_ap) =
		DedupCountTensor::compress_avx512_percent(&weights, prefix_digits, truncate_rounds);
	let (tensor_ak, sandbag_ak) =
		DedupCountTensor::compress_avx512_kl(&weights, prefix_digits, truncate_rounds);

	// Print error for all
	for (name, tensor, sandbag) in [
		("scalar_percent", &tensor_sp, &sandbag_sp),
		("scalar_kl", &tensor_sk, &sandbag_sk),
		("avx512_percent", &tensor_ap, &sandbag_ap),
		("avx512_kl", &tensor_ak, &sandbag_ak),
	] {
		let recon = tensor.decompress_all(sandbag);
		let err = mse(&weights, &recon);
		eprintln!("  {name:24} roundtrip_mse = {:.2e}", err);
	}

	// GPU quantize — bucket output (reconstruction from buckets TODO)
	if gpu_init.is_ok() {
		let gpu_out = DedupCountTensor::gpu_quantize(&weights, prefix_digits);
		if let Some(ref out) = gpu_out {
			let tails = out.last_tail_per_block();
			eprintln!("gpu_quantize last_tail_per_block: {:?}", tails);
		} else {
			eprintln!("gpu_quantize returned None despite GPU init");
		}
	}

	// Benchmark decompression only
	let mut group = c.benchmark_group("decompress");
	group.throughput(Throughput::Elements(n as u64));
	group.sample_size(256);

	group.bench_function("scalar_percent", |b| {
		b.iter(|| {
			let recon = tensor_sp.decompress_all(&sandbag_sp);
			criterion::black_box(recon);
		});
	});

	group.bench_function("scalar_kl", |b| {
		b.iter(|| {
			let recon = tensor_sk.decompress_all(&sandbag_sk);
			criterion::black_box(recon);
		});
	});

	group.bench_function("avx512_percent", |b| {
		b.iter(|| {
			let recon = tensor_ap.decompress_all(&sandbag_ap);
			criterion::black_box(recon);
		});
	});

	group.bench_function("avx512_kl", |b| {
		b.iter(|| {
			let recon = tensor_ak.decompress_all(&sandbag_ak);
			criterion::black_box(recon);
		});
	});

	group.finish();
}

criterion_group!(decompress_benches, bench_decompress);
criterion_main!(decompress_benches);

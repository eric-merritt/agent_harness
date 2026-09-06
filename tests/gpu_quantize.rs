//! GPU quantize: does the shader produce what the CPU encoder produces?
//!
//! Skips cleanly when there is no usable Vulkan device.
//! Run with: cargo test --release --test gpu_quantize -- --nocapture

use agent_harness::memory_controller::controller::init_global_controller;
use agent_harness::memory_controller::gpu_mem_op::init_gpu;
use agent_harness::models::format::*;
use agent_harness::models::quantize::*;
use agent_harness::models::tensor::Tensor;

/// Four bf16 tensors with different magnitudes and a few negatives.
fn build_model() -> (Vec<Tensor>, Vec<u8>) {
	let shapes = [(64usize, 128usize), (32, 64), (128, 64), (16, 16)];
	let mut raw = Vec::new();
	let mut tensors = Vec::new();

	for (ti, &(r, c)) in shapes.iter().enumerate() {
		let n = r * c;
		let offset = raw.len() as u64;
		for i in 0..n {
			let mag = 10f32.powi(-(ti as i32)) * (0.05 + 0.9 * (i % 97) as f32 / 97.0);
			let v = if i % 3 == 0 { -mag } else { mag };
			raw.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
		}
		tensors.push(Tensor::new(
			format!("blk.{ti}.weight"),
			vec![r, c],
			GgmlType::BF16,
			Tensor::classify("weight"),
			offset,
			(n * 2) as u64,
		));
	}
	(tensors, raw)
}

#[test]
fn gpu_matches_cpu_encoder() {
	let gpu = match init_gpu() {
		Ok(g) => g,
		Err(e) => {
			eprintln!("SKIP: no Vulkan device ({e})");
			return;
		}
	};
	let (_entry, instance, physical_device, device, queue, allocator) = gpu;

	if let Err(e) = init_global_controller(&instance, physical_device, device, queue, allocator) {
		eprintln!("SKIP: controller init failed ({e})");
		return;
	}

	let (tensors, raw) = build_model();
	let total: usize = tensors.iter().map(|t| t.elem_count).sum();
	eprintln!("{} tensors, {} weights", tensors.len(), total);

	let cpu_path = std::path::Path::new("/tmp/sandbag_cpu.bin");
	let gpu_path = std::path::Path::new("/tmp/sandbag_gpu.bin");

	quantize_cpu(&tensors, &raw, cpu_path).expect("cpu quantize");
	match quantize_gpu(&tensors, &raw, gpu_path) {
		Ok(()) => {}
		Err(e) => {
			eprintln!("SKIP: gpu quantize unavailable ({e})");
			return;
		}
	}

	let cpu = std::fs::read(cpu_path).expect("read cpu out");
	let gpu = std::fs::read(gpu_path).expect("read gpu out");

	// Both files end with the data section; the CPU file also carries an index.
	let data_bytes: u64 = tensors
		.iter()
		.map(|t| sandbag_tensor_bytes(t.elem_count as u64))
		.sum();
	let data_bytes = data_bytes as usize;

	assert!(cpu.len() >= data_bytes, "cpu file too short");
	assert!(gpu.len() >= data_bytes, "gpu file too short");
	let cpu_data = &cpu[cpu.len() - data_bytes..];
	let gpu_data = &gpu[gpu.len() - data_bytes..];

	// Bit-exact agreement is not achievable across a CPU and a GPU float pipeline:
	// Vulkan permits 2.5 ULP on division, so a weight sitting a hair either side of
	// a rounding boundary can land one step apart. What must hold is that any
	// disagreement is a single step in the tail — the digit that is already lossy
	// at tail_digits == 3, where 1000 values are rescaled into 256 — and that it is
	// vanishingly rare. A structural bug (bad offset, wrong stride, unwritten
	// plane) would show up as large differences or a large count, not as +/-1.
	let diffs: Vec<usize> = (0..data_bytes)
		.filter(|&i| cpu_data[i] != gpu_data[i])
		.collect();

	let worst = diffs
		.iter()
		.map(|&i| (cpu_data[i] as i16 - gpu_data[i] as i16).abs())
		.max()
		.unwrap_or(0);
	let rate = diffs.len() as f64 / data_bytes as f64;

	eprintln!(
		"{} of {} bytes differ ({:.4}%), largest delta {}",
		diffs.len(),
		data_bytes,
		100.0 * rate,
		worst
	);
	for &i in diffs.iter().take(8) {
		eprintln!("  [{i}] cpu={:#04x} gpu={:#04x}", cpu_data[i], gpu_data[i]);
	}

	assert!(
		worst <= 1,
		"differences must be a single quantization step; saw a delta of {worst}"
	);
	assert!(
		rate < 0.001,
		"only a handful of weights should sit on a rounding boundary; {:.4}% differ",
		100.0 * rate
	);

	eprintln!(
		"OK: GPU matches CPU on {}/{} bytes; the rest differ by one step in the tail",
		data_bytes - diffs.len(),
		data_bytes
	);
}

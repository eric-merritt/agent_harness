// compress_bench.rs

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::sync::{Arc, Mutex};

use agent_harness::memory_controller::gpu_mem_op::init_gpu;
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

fn bench_compress(c: &mut Criterion) {
	let n = 1_000_000;
	let weights = make_weights(n);

	// Unpack the entry into a named variable so its RAII lifetime keeps the library open
	let (vulkan_entry, instance, physical_device, device, queue, allocator) = 
		init_gpu().expect("Failed to boot Vulkan subsystem");

	// Pass down the regular, expected reference to the instance
	init_global_controller(
		&instance,
		physical_device,
		device.clone(),
		queue,
		allocator,
	).expect("Failed to spin up global memory controller");

	let mut group = c.benchmark_group("quantization_ops");
	
	// ── RUN 1: Measure Data Bandwidth Throughput (GB/s) ──
	let buffer_bytes = (n * std::mem::size_of::<f32>()) as u64;
	group.throughput(Throughput::Bytes(buffer_bytes));
	group.bench_function("gpu_quantize_1m_elements_bytes", |b| {
		b.iter(|| {
			let _output = DedupCountTensor::gpu_quantize(&weights, 2);
		})
	});

	// ── RUN 2: Measure Processing Element Throughput (Elem/s) ──
	group.throughput(Throughput::Elements(n as u64));
	group.bench_function("gpu_quantize_1m_elements_count", |b| {
		b.iter(|| {
			let _output = DedupCountTensor::gpu_quantize(&weights, 2);
		})
	});

	group.finish();

	// Explicitly preserve the instance to guarantee the compiler does not
	// optimize it away or drop it early during execution of the benchmark loop.
	std::mem::drop(instance);
}

criterion_group!(benches, bench_compress);
criterion_main!(benches);

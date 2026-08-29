// benches/decompress_bench.rs
//
// Decompression benchmarks with error measurement.
// Compression is done once upfront; only decompression is timed.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use agent_harness::models::dedupe::tensor::DedupCountTensor;

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

	// GPU pre-compute (uses new shader-based quantization)
	let gpu_out = agent_harness::memory_controller::gpu_mem_op::try_gpu_quantize(&weights, prefix_digits);
	if let Some(gpu_out) = &gpu_out {
		let (tensor_gp, sandbag_gp) = DedupCountTensor::compress_from_gpu_percent(
			&gpu_out.prefix_ints,
			&gpu_out.tails,
			&gpu_out.signs,
			prefix_digits,
			truncate_rounds,
		);
		let (tensor_gk, sandbag_gk) = DedupCountTensor::compress_from_gpu_kl(
			&gpu_out.prefix_ints,
			&gpu_out.tails,
			&gpu_out.signs,
			prefix_digits,
			truncate_rounds,
		);
		for (name, tensor, sandbag) in [
			("gpu_pure_percent", &tensor_gp, &sandbag_gp),
			("gpu_pure_kl", &tensor_gk, &sandbag_gk),
		] {
			let recon = tensor.decompress_all(sandbag);
			let err = mse(&weights, &recon);
			eprintln!("  {name:24} roundtrip_mse = {:.2e}", err);
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

	if let Some(gpu_out) = &gpu_out {
		let (tensor_gp, sandbag_gp) = DedupCountTensor::compress_from_gpu_percent(
			&gpu_out.prefix_ints,
			&gpu_out.tails,
			&gpu_out.signs,
			prefix_digits,
			truncate_rounds,
		);
		let (tensor_gk, sandbag_gk) = DedupCountTensor::compress_from_gpu_kl(
			&gpu_out.prefix_ints,
			&gpu_out.tails,
			&gpu_out.signs,
			prefix_digits,
			truncate_rounds,
		);

		group.bench_function("gpu_pure_percent", |b| {
			b.iter(|| {
				let recon = tensor_gp.decompress_all(&sandbag_gp);
				criterion::black_box(recon);
			});
		});

		group.bench_function("gpu_pure_kl", |b| {
			b.iter(|| {
				let recon = tensor_gk.decompress_all(&sandbag_gk);
				criterion::black_box(recon);
			});
		});
	}

	group.finish();
}

criterion_group!(decompress_benches, bench_decompress);
criterion_main!(decompress_benches);

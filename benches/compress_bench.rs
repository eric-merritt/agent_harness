//! Sandbag encode-path benchmarks.
//!
//! Synthetic weights by default. Point `SANDBAG_BENCH_MODEL` at a GGUF to bench
//! against real tensors instead:
//!
//! ```text
//! SANDBAG_BENCH_MODEL=/path/model.gguf cargo bench --bench compress_bench
//! ```

use agent_harness::models::format::*;
use agent_harness::models::quantize::*;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

/// Weights shaped like a real tensor: zero-mean, small sigma, occasional outlier.
fn synthetic_weights(n: usize) -> Vec<f32> {
	// Deterministic xorshift so runs are comparable across machines.
	let mut s = 0x2545_F491_4F6C_DD1Du64;
	let mut next = || {
		s ^= s << 13;
		s ^= s >> 7;
		s ^= s << 17;
		s
	};
	(0..n)
		.map(|i| {
			// Box-Muller from two uniforms, scaled to a typical weight sigma.
			let u1 = (next() >> 11) as f32 / (1u64 << 53) as f32;
			let u2 = (next() >> 11) as f32 / (1u64 << 53) as f32;
			let g = (-2.0 * (u1.max(1e-9)).ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
			// One in ~4096 weights is a fat outlier, as in real tensors.
			if i % 4096 == 0 { g * 0.4 } else { g * 0.02 }
		})
		.collect()
}

/// Real tensor data when a model is configured, else synthetic.
fn bench_weights(n: usize) -> Vec<f32> {
	let Ok(path) = std::env::var("SANDBAG_BENCH_MODEL") else {
		return synthetic_weights(n);
	};
	let Ok(raw) = std::fs::read(&path) else {
		eprintln!("SANDBAG_BENCH_MODEL={path} unreadable; using synthetic weights");
		return synthetic_weights(n);
	};
	let Ok(header) = parse_gguf(&raw) else {
		eprintln!("{path} is not parseable GGUF; using synthetic weights");
		return synthetic_weights(n);
	};

	let mut out = Vec::with_capacity(n);
	for t in &header.tensor_info {
		let count: u64 = t.shape.iter().product();
		let start = t.data_offset as usize;
		let end = start + t.dtype.tensor_bytes(count) as usize;
		if end > raw.len() {
			continue;
		}
		if let Ok(v) = decode_to_f32(&raw[start..end], count as usize, t.dtype) {
			out.extend(v.into_iter().take(n - out.len()));
		}
		if out.len() >= n {
			break;
		}
	}
	if out.len() < n {
		out.extend(synthetic_weights(n - out.len()));
	}
	out
}

const N: usize = 1 << 20; // 1M weights per iteration

fn bench_encode_tail_digits(c: &mut Criterion) {
	let vals = bench_weights(N);
	let threshold = calibrate(&vals, ScaleMethod::MaxAbs);

	let mut g = c.benchmark_group("encode/tail_digits");
	g.throughput(Throughput::Elements(N as u64));
	for digits in [0u32, 1, 2, 3] {
		g.bench_with_input(BenchmarkId::from_parameter(digits), &digits, |b, &d| {
			b.iter(|| encode_sandbag(black_box(&vals), d, threshold));
		});
	}
	g.finish();
}

fn bench_calibration(c: &mut Criterion) {
	let vals = bench_weights(N);

	let mut g = c.benchmark_group("calibrate");
	g.throughput(Throughput::Elements(N as u64));
	for (name, method) in [
		("max_abs", ScaleMethod::MaxAbs),
		("percentile_99.9", ScaleMethod::Percentile(99.9)),
		("kl_divergence", ScaleMethod::KlDivergence),
	] {
		g.bench_function(name, |b| {
			b.iter(|| calibrate(black_box(&vals), method));
		});
	}
	g.finish();
}

fn bench_decode_source(c: &mut Criterion) {
	let vals = synthetic_weights(N);

	let f32_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
	let bf16_bytes: Vec<u8> = vals
		.iter()
		.flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
		.collect();
	let f16_bytes: Vec<u8> = vals
		.iter()
		.flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
		.collect();

	let mut g = c.benchmark_group("decode_source");
	g.throughput(Throughput::Elements(N as u64));
	for (name, bytes, ty) in [
		("f32", &f32_bytes, GgmlType::F32),
		("bf16", &bf16_bytes, GgmlType::BF16),
		("f16", &f16_bytes, GgmlType::F16),
	] {
		g.bench_function(name, |b| {
			b.iter(|| decode_to_f32(black_box(bytes), N, ty).unwrap());
		});
	}
	g.finish();
}

/// The whole per-tensor path: widen, calibrate, encode.
fn bench_end_to_end(c: &mut Criterion) {
	let vals = synthetic_weights(N);
	let bf16: Vec<u8> = vals
		.iter()
		.flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
		.collect();

	let mut g = c.benchmark_group("tensor_end_to_end");
	g.throughput(Throughput::Elements(N as u64));
	g.bench_function("bf16_to_sandbag", |b| {
		b.iter(|| {
			let v = decode_to_f32(black_box(&bf16), N, GgmlType::BF16).unwrap();
			let t = calibrate(&v, ScaleMethod::MaxAbs);
			encode_sandbag(&v, DEFAULT_TAIL_DIGITS, t)
		});
	});
	g.finish();
}

criterion_group!(
	benches,
	bench_encode_tail_digits,
	bench_calibration,
	bench_decode_source,
	bench_end_to_end
);
criterion_main!(benches);

//! 4-bit group-quantized GEMV: the inference-side read path.
//!
//! Kept as the reference point sandbag has to beat. `gemv_4bit_into` dequantizes
//! from packed nibbles inside the matvec rather than materializing f32 weights,
//! which is the same shape as the intended sandbag kernel — decode inline, no
//! decompression pass.

use agent_harness::kernels::gemv::{gemv, gemv_4bit_into, gemv_4bit_scalar, quantize};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

const GROUP_SIZE: usize = 32;

fn sample(n: usize, seed: u64) -> Vec<f32> {
	let mut s = seed | 1;
	(0..n)
		.map(|_| {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			((s >> 40) as f32 / 16_777_216.0 - 0.5) * 0.08
		})
		.collect()
}

fn bench_quantize(c: &mut Criterion) {
	let mut g = c.benchmark_group("quantize_4bit");
	for &n in &[1 << 16, 1 << 20] {
		let w = sample(n, 0xABCD);
		g.throughput(Throughput::Elements(n as u64));
		g.bench_with_input(BenchmarkId::from_parameter(n), &w, |b, w| {
			b.iter(|| quantize(black_box(w), GROUP_SIZE));
		});
	}
	g.finish();
}

fn bench_gemv(c: &mut Criterion) {
	// A square-ish projection typical of a small model's attention block.
	let (rows, cols) = (2048usize, 2048usize);
	let w = sample(rows * cols, 0x1357);
	let x = sample(cols, 0x2468);
	let (scales, packed) = quantize(&w, GROUP_SIZE);
	let mut out = vec![0.0f32; rows];

	let mut g = c.benchmark_group("gemv_2048x2048");
	g.throughput(Throughput::Elements((rows * cols) as u64));

	g.bench_function("f32_dense", |b| {
		b.iter(|| gemv(black_box(&w), black_box(&x), rows, cols));
	});
	g.bench_function("4bit_scalar", |b| {
		b.iter(|| {
			// Takes a row range, not a row count.
			gemv_4bit_scalar(
				&mut out,
				black_box(&scales),
				black_box(&packed),
				black_box(&x),
				0,
				rows,
				cols,
				GROUP_SIZE,
			)
		});
	});
	g.bench_function("4bit_dispatch", |b| {
		b.iter(|| {
			gemv_4bit_into(
				&mut out,
				black_box(&scales),
				black_box(&packed),
				black_box(&x),
				rows,
				cols,
				GROUP_SIZE,
			)
		});
	});
	g.finish();
}

criterion_group!(benches, bench_quantize, bench_gemv);
criterion_main!(benches);

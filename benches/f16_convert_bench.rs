//! bf16/f16 -> f32 widening: scalar vs the AVX-512 kernels.
//!
//! This is the first stage of every conversion, so it sets the ceiling on
//! `quantize_cpu` throughput. `models::quantize::decode_to_f32` currently takes
//! the scalar path even though `kernels::avx512` has vectorized versions —
//! this bench is what that decision should be judged on.

use agent_harness::kernels::avx512::{avx512_bf16_to_f32, avx512_f16_to_f32};
use agent_harness::models::format::GgmlType;
use agent_harness::models::quantize::decode_to_f32;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

const N: usize = 1 << 20;

fn sample_f32(n: usize) -> Vec<f32> {
	let mut s = 0x1234_5678_9ABC_DEF0u64;
	(0..n)
		.map(|_| {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			// Spread across a realistic weight range rather than [0,1).
			((s >> 40) as f32 / 16_777_216.0 - 0.5) * 0.08
		})
		.collect()
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
	vals.iter()
		.flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
		.collect()
}

fn f16_bytes(vals: &[f32]) -> Vec<u8> {
	vals.iter()
		.flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
		.collect()
}

fn has_avx512() -> bool {
	#[cfg(target_arch = "x86_64")]
	{
		is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		false
	}
}

fn bench_bf16(c: &mut Criterion) {
	let vals = sample_f32(N);
	let src = bf16_bytes(&vals);
	let mut out = vec![0.0f32; N];

	let mut g = c.benchmark_group("bf16_to_f32");
	g.throughput(Throughput::Elements(N as u64));

	g.bench_function("scalar_decode_to_f32", |b| {
		b.iter(|| decode_to_f32(black_box(&src), N, GgmlType::BF16).unwrap());
	});

	if has_avx512() {
		g.bench_function("avx512", |b| {
			b.iter(|| unsafe { avx512_bf16_to_f32(black_box(&src), &mut out) });
		});
	} else {
		eprintln!("avx512f/avx512bw not present; skipping vectorized bf16 bench");
	}
	g.finish();
}

fn bench_f16(c: &mut Criterion) {
	let vals = sample_f32(N);
	let src = f16_bytes(&vals);
	let mut out = vec![0.0f32; N];

	let mut g = c.benchmark_group("f16_to_f32");
	g.throughput(Throughput::Elements(N as u64));

	g.bench_function("scalar_decode_to_f32", |b| {
		b.iter(|| decode_to_f32(black_box(&src), N, GgmlType::F16).unwrap());
	});

	if has_avx512() {
		g.bench_function("avx512", |b| {
			b.iter(|| unsafe { avx512_f16_to_f32(black_box(&src), &mut out) });
		});
	} else {
		eprintln!("avx512f/avx512bw not present; skipping vectorized f16 bench");
	}
	g.finish();
}

criterion_group!(benches, bench_bf16, bench_f16);
criterion_main!(benches);

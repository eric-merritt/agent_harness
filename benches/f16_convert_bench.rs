// benches/f16_convert_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use agent_harness::models::avx512_kernel::{
    avx512_bf16_to_f32, avx512_f16_to_f32,
    bf16_to_f32_scalar, dispatch_bf16_bytes_to_f32, dispatch_f16_bytes_to_f32,
    f16_to_f32_scalar,
};

/// Deterministic pseudo-random byte buffer (LCG, no rand crate needed).
fn make_bytes(n_bytes: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(n_bytes);
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    for _ in 0..(n_bytes / 2) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 32) as u16;
        buf.push(bits as u8);
        buf.push((bits >> 8) as u8);
    }
    buf
}

/// Pure scalar F16 -> f32 (element-by-element, no SIMD).
fn f16_scalar_loop(src: &[u8]) -> Vec<f32> {
    let n = src.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        out.push(f16_to_f32_scalar(bits));
    }
    out
}

/// Pure scalar BF16 -> f32 (element-by-element, no SIMD).
fn bf16_scalar_loop(src: &[u8]) -> Vec<f32> {
    let n = src.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        out.push(bf16_to_f32_scalar(bits));
    }
    out
}

fn bench_f16(c: &mut Criterion) {
    let n_elems = 4_096_000; // ~8 MB
    let src = make_bytes(n_elems * 2);

    let mut group = c.benchmark_group("f16_to_f32");
    group.throughput(Throughput::Elements(n_elems as u64));

    group.bench_function("scalar_loop", |b| {
        b.iter(|| {
            let out = f16_scalar_loop(&src);
            criterion::black_box(&out);
        });
    });

    if is_x86_feature_detected!("avx512f") {
        group.bench_function("avx512_dispatch", |b| {
            b.iter(|| {
                let out = dispatch_f16_bytes_to_f32(&src);
                criterion::black_box(&out);
            });
        });

        group.bench_function("avx512_kernel_only", |b| {
            b.iter(|| {
                let mut out = vec![0f32; n_elems];
                unsafe { avx512_f16_to_f32(&src, &mut out); }
                criterion::black_box(&out);
            });
        });
    }

    group.finish();
}

fn bench_bf16(c: &mut Criterion) {
    let n_elems = 4_096_000;
    let src = make_bytes(n_elems * 2);

    let mut group = c.benchmark_group("bf16_to_f32");
    group.throughput(Throughput::Elements(n_elems as u64));

    group.bench_function("scalar_loop", |b| {
        b.iter(|| {
            let out = bf16_scalar_loop(&src);
            criterion::black_box(&out);
        });
    });

    if is_x86_feature_detected!("avx512f") {
        group.bench_function("avx512_dispatch", |b| {
            b.iter(|| {
                let out = dispatch_bf16_bytes_to_f32(&src);
                criterion::black_box(&out);
            });
        });

        group.bench_function("avx512_kernel_only", |b| {
            b.iter(|| {
                let mut out = vec![0f32; n_elems];
                unsafe { avx512_bf16_to_f32(&src, &mut out); }
                criterion::black_box(&out);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_f16, bench_bf16);
criterion_main!(benches);

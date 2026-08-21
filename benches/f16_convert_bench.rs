// benches/f16_convert_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use agent_harness::models::avx512_kernel::{
    bf16_bytes_to_f32, f16_bytes_to_f32,
};

/// Build a deterministic pseudo-random byte buffer (no rand dependency needed).
fn make_bytes(n_bytes: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(n_bytes);
    // Simple LCG so the bench is reproducible without a rand crate.
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    for _ in 0..(n_bytes / 2) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (state >> 32) as u16;
        buf.push(bits as u8);
        buf.push((bits >> 8) as u8);
    }
    buf
}

fn bench_f16(c: &mut Criterion) {
    let n_elems = 4_096_000; // ~8 MB of f16
    let src = make_bytes(n_elems * 2);

    let mut group = c.benchmark_group("f16_to_f32");
    group.throughput(Throughput::Elements(n_elems as u64));

    group.bench_function("bytes_to_f32", |b| {
        b.iter(|| {
            let out = f16_bytes_to_f32(&src);
            criterion::black_box(&out);
        });
    });

    group.finish();
}

fn bench_bf16(c: &mut Criterion) {
    let n_elems = 4_096_000;
    let src = make_bytes(n_elems * 2);

    let mut group = c.benchmark_group("bf16_to_f32");
    group.throughput(Throughput::Elements(n_elems as u64));

    group.bench_function("bytes_to_f32", |b| {
        b.iter(|| {
            let out = bf16_bytes_to_f32(&src);
            criterion::black_box(&out);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_f16, bench_bf16);
criterion_main!(benches);

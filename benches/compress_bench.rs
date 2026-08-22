// benches/compress_bench.rs
//
// Compares the three compression methods for a 1M-element chunk:
//   - compress_scalar  (no SIMD)
//   - compress_avx512  (AVX-512 kernel + scalar fallback)
//   - compress_gpu     (wgpu per-element pass + CPU dedup)
//
// Scalar/AVX-512 are always run; the GPU bench is skipped (with a printed
// notice) when no GPU adapter is available.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use agent_harness::models::dedup::tensor::DedupCountTensor;

/// Deterministic pseudo-random weights (LCG). Small values (~N(0,0.3)-ish
/// spread) so the tensor doesn't trip the full-precision escape.
fn make_weights(n: usize) -> Vec<f32> {
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut out = Vec::with_capacity(n);
    for _i in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to roughly [-1.0, 1.0), denser near zero
        let u = ((state >> 40) as f64) / (1u64 << 24) as f64;
        let v = ((state >> 20) as f64) / (1u64 << 24) as f64;
        let x = (u - 0.5).sqrt().signum() * ((u - 0.5).abs().sqrt() + v * 0.5) * 2.0;
        out.push(x as f32);
    }
    out
}

fn bench_compress(c: &mut Criterion) {
    let n = 1_000_000; // 4 MB — matches CHUNK_SIZE used by the convert pipeline
    let prefix_digits = 2;
    let truncate_rounds = 2;
    let weights = make_weights(n);

    let mut group = c.benchmark_group("dedup_compress");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("scalar", |b| {
        b.iter(|| {
            let (t, m) = DedupCountTensor::compress_scalar(&weights, prefix_digits, truncate_rounds);
            criterion::black_box((&t, &m));
        });
    });

    group.bench_function("avx512_dispatch", |b| {
        b.iter(|| {
            let (t, m) = DedupCountTensor::compress_avx512(&weights, prefix_digits, truncate_rounds);
            criterion::black_box((&t, &m));
        });
    });

    if !is_x86_feature_detected!("avx512f") {
        eprintln!("compress_bench: AVX-512 not supported on this CPU — avx512_dispatch falls back to scalar internally.");
    }

    // GPU paths: skipped entirely if no adapter, so CI without a GPU stays green.
    let gpu_out = agent_harness::gpu::gpu_compute(&weights, prefix_digits);
    if let Some(gpu_out) = &gpu_out {
        let prefix_bits = gpu_out.prefix_bits.clone();
        let tails = gpu_out.tails.clone();
        let signs = gpu_out.signs.clone();

        group.bench_function("gpu_scalar_tails", |b| {
            b.iter(|| {
                let (t, m) = DedupCountTensor::compress_from_gpu(&weights, &prefix_bits, &tails, &signs, prefix_digits, truncate_rounds);
                criterion::black_box((&t, &m));
            });
        });

        group.bench_function("gpu_with_avx512_tails", |b| {
            b.iter(|| {
                let (t, m) = DedupCountTensor::compress_gpu_with_avx512(&weights, &prefix_bits, &tails, &signs, prefix_digits, truncate_rounds);
                criterion::black_box((&t, &m));
            });
        });
    } else {
        eprintln!("compress_bench: no GPU adapter available — skipping gpu benches.");
    }

    group.finish();
}

criterion_group!(benches, bench_compress);
criterion_main!(benches);

// benches/compress_bench.rs
//
// Compression benchmarks only — no decompression, no error calculation.
// Measures raw compression throughput.

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

fn bench_compress(c: &mut Criterion) {
    let n = 1_000_000;
    let prefix_digits = 2;
    let truncate_rounds = 3;
    let weights = make_weights(n);

    let mut group = c.benchmark_group("compress");
    group.throughput(Throughput::Elements(n as u64));
    group.sample_size(50);

    // ── Scalar percentile ────────────────────────────────────────────────
    group.bench_function("scalar_percent", |b| {
        b.iter(|| {
            let (t, m) =
                DedupCountTensor::compress_quantized(&weights, prefix_digits, truncate_rounds);
            criterion::black_box((&t, &m));
        });
    });

    // ── Scalar KL ────────────────────────────────────────────────────────
    group.bench_function("scalar_kl", |b| {
        b.iter(|| {
            let (t, m) =
                DedupCountTensor::compress_quantized_kl(&weights, prefix_digits, truncate_rounds);
            criterion::black_box((&t, &m));
        });
    });

    // ── AVX-512 percentile ───────────────────────────────────────────────
    group.bench_function("avx512_percent", |b| {
        b.iter(|| {
            let (t, m) =
                DedupCountTensor::compress_avx512_percent(&weights, prefix_digits, truncate_rounds);
            criterion::black_box((&t, &m));
        });
    });

    // ── AVX-512 KL ───────────────────────────────────────────────────────
    group.bench_function("avx512_kl", |b| {
        b.iter(|| {
            let (t, m) =
                DedupCountTensor::compress_avx512_kl(&weights, prefix_digits, truncate_rounds);
            criterion::black_box((&t, &m));
        });
    });

    // GPU pre-compute (shared by all GPU paths) — ONCE before all GPU benchmarks
    let gpu_out = agent_harness::gpu::gpu_compute(&weights, prefix_digits);

    if let Some(gpu_out) = &gpu_out {
        let prefix_ints = gpu_out.prefix_ints.clone();
        let tails = gpu_out.tails.clone();
        let signs = gpu_out.signs.clone();

        // ── GPU pure percentile ────────────────────────────────────────────
        group.bench_function("gpu_pure_percent", |b| {
            b.iter(|| {
                let (t, m) = DedupCountTensor::compress_from_gpu_percent(
                    &gpu_out.prefix_ints,
                    &tails,
                    &signs,
                    prefix_digits,
                    truncate_rounds,
                );
                criterion::black_box((&t, &m));
            });
        });

        // ── GPU pure KL ────────────────────────────────────────────────────
        group.bench_function("gpu_pure_kl", |b| {
            b.iter(|| {
                let (t, m) = DedupCountTensor::compress_from_gpu_kl(
                    &gpu_out.prefix_ints,
                    &tails,
                    &signs,
                    prefix_digits,
                    truncate_rounds,
                );
                criterion::black_box((&t, &m));
            });
        });

        // ── GPU + AVX512 tails percentile ──────────────────────────────────
        group.bench_function("gpu_avx512_tails_percent", |b| {
            b.iter(|| {
                let (t, m) = DedupCountTensor::compress_from_gpu_percent(
                    &prefix_ints,
                    &tails,
                    &signs,
                    prefix_digits,
                    truncate_rounds,
                );
                criterion::black_box((&t, &m));
            });
        });

        // ── GPU + AVX512 tails KL ──────────────────────────────────────────
        group.bench_function("gpu_avx512_tails_kl", |b| {
            b.iter(|| {
                let (t, m) = DedupCountTensor::compress_from_gpu_kl(
                    &prefix_ints,
                    &tails,
                    &signs,
                    prefix_digits,
                    truncate_rounds,
                );
                criterion::black_box((&t, &m));
            });
        });

        // ── GPU + scalar tails percentile ──────────────────────────────────
        group.bench_function("gpu_scalar_tails_percent", |b| {
            b.iter(|| {
                let (t, m) = DedupCountTensor::compress_from_gpu_scalar_percent(
                    &prefix_ints,
                    &tails,
                    &signs,
                    prefix_digits,
                    truncate_rounds,
                );
                criterion::black_box((&t, &m));
            });
        });

        // ── GPU + scalar tails KL ──────────────────────────────────────────
        group.bench_function("gpu_scalar_tails_kl", |b| {
            b.iter(|| {
                let (t, m) = DedupCountTensor::compress_from_gpu_scalar_kl(
                    &prefix_ints,
                    &tails,
                    &signs,
                    prefix_digits,
                    truncate_rounds,
                );
                criterion::black_box((&t, &m));
            });
        });
    } else {
        eprintln!("compress_bench: no GPU adapter available — skipping GPU benches.");
    }

    if !is_x86_feature_detected!("avx512f") {
        eprintln!(
            "compress_bench: AVX-512 not supported on this CPU — avx512 methods fall back to scalar internally."
        );
    }

    group.finish();
}

criterion_group!(compress_benches, bench_compress);
criterion_main!(compress_benches);

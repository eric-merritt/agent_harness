// benches/dequant_bench.rs
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

// Clean decoupled path imports matching your layout
use agent_harness::inference::math::{gemv_4bit_into, gemv_4bit_parallel_worker_scalar};
use agent_harness::models::quantization::{GROUP_SIZE, quantize};

pub fn bench_gemv_4bit(c: &mut Criterion) {
    let mut group = c.benchmark_group("GEMV_4Bit_INT4");
    let rows = 4096;
    let cols = 4096;

    let mock_x = vec![0.5f32; cols];
    let mock_weights = vec![0.35f32; rows * cols];

    let (scales, packed) = quantize(&mock_weights, GROUP_SIZE);
    let mut output = vec![0.0f32; rows];

    group.throughput(Throughput::Elements((rows * cols) as u64));

    group.bench_function(BenchmarkId::new("Scalar_GEMV", rows * cols), |b| {
        b.iter(|| {
            gemv_4bit_parallel_worker_scalar(
                &mut output,
                &scales,
                &packed,
                &mock_x,
                0,
                rows,
                cols,
                GROUP_SIZE,
            );
            criterion::black_box(&output);
        });
    });

    group.bench_function(BenchmarkId::new("Fused_AVX-512_GEMV", rows * cols), |b| {
        b.iter(|| {
            gemv_4bit_into(
                &mut output,
                &scales,
                &packed,
                &mock_x,
                rows,
                cols,
                GROUP_SIZE,
            );
            criterion::black_box(&output);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_gemv_4bit);
criterion_main!(benches);

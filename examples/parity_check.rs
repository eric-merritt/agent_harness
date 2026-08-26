use crate::main::dedupe::tensor::DedupCountTensor::{compress_from_gpu, compress_gpu_w_avx512};

/// Quick parity check: pure-GPU path vs gpu_with_avx512 path

fn main() {
    let n = 1_000_000;
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut weights = Vec::with_capacity(n);
    for _i in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((state >> 40) as f64) / (1u64 << 24) as f64;
        let v = ((state >> 20) as f64) / (1u64 << 24) as f64;
        let x = (u - 0.5).sqrt().signum() * ((u - 0.5).abs().sqrt() + v * 0.5) * 2.0;
        weights.push(x as f32);
    }

    let prefix_digits = 2usize;
    let truncate_rounds = 2usize;

    let out = agent_harness::gpu::gpu_compute(&weights, prefix_digits).expect("GPU available");
    let t0 = std::time::Instant::now();
    let (t_pure, m_pure) = agent_harness::models::dedupe::tensor::DedupCountTensor::compress_from_gpu(
        &out.prefix_ints, &out.tails, &out.signs, prefix_digits, truncate_rounds);
    let pure_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = std::time::Instant::now();
    let (t_avx, m_avx) = agent_harness::models::dedupe::tensor::DedupCountTensor::compress_gpu_w_avx512(
        &weights, &out.prefix_bits, &out.tails, &out.signs, prefix_digits, truncate_rounds);
    let avx_ms = t1.elapsed().as_secs_f64() * 1e3;

    println!("pure-gpu: {:.3} ms | avx512: {:.3} ms", pure_ms, avx_ms);
    println!("prefixes equal: {}", t_pure.prefixes == t_avx.prefixes);
    println!("unique_tails equal: {}", t_pure.unique_tails == t_avx.unique_tails);
    println!("prefix_counts equal: {}", t_pure.prefix_counts == t_avx.prefix_counts);
    println!("count: {} vs {}", t_pure.count, t_avx.count);
    println!("avg_precision_lost: {} vs {}", t_pure.avg_precision_lost, t_avx.avg_precision_lost);

    // Compare sandbags element-wise
    let mut tail_diff = 0;
    for i in 0..n {
        if m_pure.tail_idx[i] != m_avx.tail_idx[i] { tail_diff += 1; }
        if m_pure.prefix_idx[i] != m_avx.prefix_idx[i] { tail_diff += 1; }
    }
    let mut sign_diff = 0;
    for i in 0..((n + 7) / 8) {
        if m_pure.sign_bits[i] != m_avx.sign_bits[i] { sign_diff += 1; }
    }
    println!("tail_idx/prefix_idx diffs: {}  sign_bits byte diffs: {}", tail_diff, sign_diff);

    // Round-trip: decompress both and compare
    let d_pure = t_pure.decompress_all(&m_pure);
    let d_avx = t_avx.decompress_all(&m_avx);
    let mut max_diff = 0.0f32;
    for i in 0..n {
        let d = (d_pure[i] - d_avx[i]).abs();
        if d > max_diff { max_diff = d; }
    }
    println!("max decompress diff: {}", max_diff);
}

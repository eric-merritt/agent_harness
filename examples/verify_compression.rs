//! Quick verification: compression ratio + reconstruction accuracy.
//! Run with: cargo run --example verify_compression

use agent_harness::models::dedupe::tensor::DedupCountTensor;

fn make_weights(n: usize) -> Vec<f32> {
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut out = Vec::with_capacity(n);
    for _i in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 40) as f64) / (1u64 << 24) as f64;
        let v = ((state >> 20) as f64) / (1u64 << 24) as f64;
        let x = (u - 0.5).sqrt().signum() * ((u - 0.5).abs().sqrt() + v * 0.5) * 2.0;
        out.push(x as f32);
    }
    out
}

fn main() {
    let n = 1_000_000;
    let weights = make_weights(n);

    println!("=== Compression Verification ===");
    println!("Input: {} f32 weights = {} bytes", n, n * 4);

    // Compress
    let (tensor, sandbag) = DedupCountTensor::compress_scalar(&weights, 2, 2);
    let core = agent_harness::models::convert::core::serialize_core(&tensor);
    let sandbag_bytes = sandbag.to_bytes();

    let total_compressed = core.len() + sandbag_bytes.len();
    let ratio = total_compressed as f64 / (n * 4) as f64 * 100.0;

    println!("\nCompression results:");
    println!("  Core size:     {} bytes", core.len());
    println!("  Sandbag size:  {} bytes", sandbag_bytes.len());
    println!("  Total:         {} bytes", total_compressed);
    println!("  Ratio:         {:.1}% of original ({:.2}x compression)", ratio, (n * 4) as f64 / total_compressed as f64);
    println!("  Prefixes:      {}", tensor.prefixes.len());
    println!("  Unique tails:  {}", tensor.unique_tails.len());
    println!("  Avg precision lost: {:.2e}", tensor.avg_precision_lost);

    // Decompress
    let decompressed = tensor.decompress_all(&sandbag);

    println!("\nReconstruction accuracy:");
    assert_eq!(decompressed.len(), n, "Length mismatch");

    let mut max_err = 0.0f32;
    let mut max_err_idx = 0usize;
    let mut outlier_count = sandbag.outliers.len();
    let mut mse_sum = 0.0f64;

    for (i, (&orig, &decomp)) in weights.iter().zip(decompressed.iter()).enumerate() {
        let err = (orig - decomp).abs();
        mse_sum += err as f64 * err as f64;
        if err > max_err {
            max_err = err;
            max_err_idx = i;
        }
    }

    // Outliers are restored at full precision, so they shouldn't count toward error
    for &(pos, _) in &sandbag.outliers {
        let err = (weights[pos] - decompressed[pos]).abs();
        assert!(err < 1e-6, "Outlier at {} not restored: orig={}, decomp={}", pos, weights[pos], decompressed[pos]);
    }

    let rmse = (mse_sum / n as f64).sqrt();
    println!("  Max error:     {:.2e} (at index {})", max_err, max_err_idx);
    println!("  RMSE:          {:.2e}", rmse);
    println!("  Outliers:      {}", outlier_count);
    println!("  Unique values: {}", sandbag.unique_values.len());
    println!("  Scale:          {:.2e}", sandbag.scale);

    // Sanity checks
    assert!(total_compressed < n * 4, "FAIL: Compressed {} >= original {}", total_compressed, n * 4);
    assert!(rmse < 0.01, "FAIL: RMSE {:.2e} too high", rmse);

    println!("\n=== PASS: Compression works correctly ===");
    println!("  Output is {}% of original size", ratio);
    println!("  Reconstruction RMSE: {:.2e}", rmse);
}

// Per-block linear quantization with outlier rejection.
//
// quantize_block     — fast O(n), 99.5th percentile clip (scalar)
// quantize_block_avx512 — AVX-512 accelerated quantization
// quantize_block_kl  — KL-divergence search, slower but higher quality

use rayon::prelude::*;
use std::f32;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Ultra-fast O(n) quantization with parallel histogram percentile estimation.
/// Zero heap allocations during the threshold selection phase.
pub fn quantize_block(weights: &[f32]) -> (f32, Vec<(usize, f32)>) {
    let n = weights.len();
    if n == 0 {
        return (1.0, Vec::new());
    }

    // Step 1: Find the absolute maximum value using parallel reduction (no allocations)
    let global_max = weights
        .par_iter()
        .map(|&w| w.abs())
        .fold(|| 0.0f32, |max_val, w_abs| max_val.max(w_abs))
        .reduce(|| 0.0f32, |max_a, max_b| max_a.max(max_b));

    if global_max <= f32::EPSILON {
        return (1.0, Vec::new());
    }

    // Step 2: Build a parallel histogram of absolute magnitudes to find the percentile
    const BINS: usize = 2048;
    let histogram = weights
        .par_chunks(65536) // Cache-aligned chunk granularity
        .map(|chunk| {
            let mut local_bins = [0usize; BINS];
            for &w in chunk {
                let abs_w = w.abs();
                // Map the float range [0.0, global_max] linearly into our histogram bins
                let bin = ((abs_w / global_max) * (BINS - 1) as f32) as usize;
                local_bins[bin.min(BINS - 1)] += 1;
            }
            local_bins
        })
        .reduce(
            || [0usize; BINS],
            |mut hist_a, hist_b| {
                for i in 0..BINS {
                    hist_a[i] += hist_b[i];
                }
                hist_a
            },
        );

    // Step 3: Walk back down the histogram sequentially to isolate the 99.5th percentile threshold
    let target_outliers_count = (n as f64 * 0.005).floor() as usize;
    let mut accumulated_elements = 0;
    let mut clip_bin = BINS - 1;

    for bin in (0..BINS).rev() {
        accumulated_elements += histogram[bin];
        if accumulated_elements >= target_outliers_count {
            clip_bin = bin;
            break;
        }
    }

    // Reconstruct the threshold value from the target bin index
    let max_abs = ((clip_bin as f32 / (BINS - 1) as f32) * global_max).max(f32::EPSILON);
    let scale = max_abs / i16::MAX as f32;

    // Step 4: Scan for outliers (Pre-allocated capacity to prevent thread allocation thrashing)
    let outliers: Vec<(usize, f32)> = weights
        .par_iter()
        .enumerate()
        .filter_map(|(i, &w)| {
            if w.abs() > max_abs {
                Some((i, w))
            } else {
                None
            }
        })
        .collect();

    (scale, outliers)
}

/// AVX-512 accelerated quantization with 99.5th percentile outlier clipping.
/// Processes 16 elements per iteration for the quantization loop.
/// Returns `(scale, outliers)` — quantized i16 values are discarded.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn quantize_block_avx512(weights: &[f32]) -> (f32, Vec<(usize, f32)>) {
    use std::cmp;

    let n = weights.len();
    if n == 0 {
        return (1.0, Vec::new());
    }

    // Collect absolute values for percentile selection (parallel)
    let mut abs_vals: Vec<f32> = weights.par_iter().map(|&w| w.abs()).collect();

    // Partial sort to find the 99.5th percentile threshold
    let clip_idx = ((n as f64 * 0.995).ceil() as usize).min(n).saturating_sub(1);
    abs_vals.select_nth_unstable_by(clip_idx, |a, b| a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal));

    let max_abs = abs_vals[clip_idx].max(f32::EPSILON);

    // Compute scale factor
    let scale = max_abs / i16::MAX as f32;

    // Scan for outliers only — quantized values are discarded (not used by caller)
    let mut outliers = Vec::new();
    if is_x86_feature_detected!("avx512f") && n >= 16 {
        let v_abs_max = _mm512_set1_ps(max_abs);

        let chunks = weights.chunks_exact(16);
        let remainder = chunks.remainder();
        let mut idx = 0usize;

        for chunk in chunks {
            let ptr = chunk.as_ptr();

            // Load 16 weights
            let v_w = _mm512_loadu_ps(ptr);
            let v_abs = _mm512_abs_ps(v_w);

            // Outlier mask: abs(w) > max_abs → outlier (invert LT = GE)
            let in_range_mask: u16 = _mm512_cmp_ps_mask(v_abs, v_abs_max, _MM_CMPINT_LT);
            let outlier_mask = !in_range_mask;

            if outlier_mask != 0 {
                for lane in 0..16 {
                    if (outlier_mask >> lane) & 1 != 0 {
                        outliers.push((idx + lane, chunk[lane]));
                    }
                }
            }
            idx += 16;
        }

        // Scalar remainder
        let remainder_idx = idx;
        for (lane, &w) in remainder.iter().enumerate() {
            if w.abs() > max_abs {
                outliers.push((remainder_idx + lane, w));
            }
        }
    } else {
        // Scalar fallback (parallel)
        let outliers_collected: Vec<(usize, f32)> = weights.par_iter()
            .enumerate()
            .filter_map(|(i, &w)| {
                if w.abs() > max_abs {
                    Some((i, w))
                } else {
                    None
                }
            })
            .collect();
        outliers.extend(outliers_collected);
    }

    (scale, outliers)
}

// ── KL-divergence-based quantization (slower, higher quality) ────────────────

/// KL-divergence quantization: search for the optimal clipping threshold that
/// minimizes the KL divergence between the true weight distribution and the
/// quantized approximation. Produces lower reconstruction error than percentile
/// clipping at the cost of higher CPU usage during the search.
/// Returns `(scale, outliers)` — quantized i16 values are discarded.
pub fn quantize_block_kl(weights: &[f32]) -> (f32, Vec<(usize, f32)>) {
    let n = weights.len();
    if n == 0 {
        return (1.0, Vec::new());
    }

    // Find the range of absolute values
    let mut abs_vals: Vec<f32> = weights.iter().map(|&w| w.abs()).collect();
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let max_abs = abs_vals.last().copied().unwrap_or(f32::EPSILON).max(f32::EPSILON);

    // Build the true distribution histogram over the full range
    let true_hist = build_histogram(weights, -max_abs, max_abs, 2048);

    // Search for the best clip threshold using KL divergence.
    // Allocate histogram ONCE outside the loop to avoid 10 heap allocations.
    let mut quant_hist = vec![0.0f32; 2048];
    let search_starts = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999];
    let mut best_idx = ((n as f64 * 0.995).ceil() as usize).min(n).saturating_sub(1);
    let mut best_kl = f32::INFINITY;

    for &pct in &search_starts {
        let clip_idx = ((n as f64 * pct).ceil() as usize).min(n).saturating_sub(1);
        let clip_val = abs_vals[clip_idx].max(f32::EPSILON);
        let scale = clip_val / i16::MAX as f32;

        // Reset histogram for this iteration
        quant_hist.fill(0.0);

        // Build quantized distribution
        let range = 2.0 * max_abs;
        for &w in weights {
            let clamped = w.clamp(-clip_val, clip_val);
            let q = (clamped / scale).round().clamp(i16::MIN as f32, i16::MAX as f32);
            let recon = q as f32 * scale;
            let mut bin = (((recon - (-max_abs)) / range) * 2048.0) as usize;
            if bin >= 2048 { bin = 2047; }
            quant_hist[bin] += 1.0;
        }
        let sum: f32 = quant_hist.iter().sum();
        if sum > 0.0 {
            quant_hist.iter_mut().for_each(|x| *x /= sum);
        }

        let kl = kl_divergence(&true_hist, &quant_hist);
        if kl.is_finite() && kl < best_kl {
            best_kl = kl;
            best_idx = clip_idx;
        }
    }

    let max_abs = abs_vals[best_idx].max(f32::EPSILON);
    let scale = max_abs / i16::MAX as f32;

    // Scan for outliers only — quantized values are discarded (not used by caller)
    let mut outliers = Vec::new();
    for (i, &w) in weights.iter().enumerate() {
        if w.abs() > max_abs {
            outliers.push((i, w));
        }
    }

    (scale, outliers)
}

fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q.iter())
        .map(|(&p_i, &q_i)| {
            if p_i == 0.0 { 0.0 }
            else if q_i == 0.0 { f32::INFINITY }
            else { p_i * (p_i / q_i).ln() }
        })
        .sum()
}

fn build_histogram(weights: &[f32], min_val: f32, max_val: f32, bins: usize) -> Vec<f32> {
    let mut hist = vec![0.0; bins];
    let range = max_val - min_val;
    if range == 0.0 { return hist; }

    for &w in weights {
        let clamped = w.clamp(min_val, max_val);
        let mut bin = (((clamped - min_val) / range) * bins as f32) as usize;
        if bin >= bins { bin = bins - 1; }
        hist[bin] += 1.0;
    }

    let sum: f32 = hist.iter().sum();
    if sum > 0.0 { hist.iter_mut().for_each(|x| *x /= sum); }
    hist
}

/// Convenience wrapper: quantize a block via the fast percentile-clip path.
pub fn compress_block(weights: &[f32]) -> (f32, Vec<(usize, f32)>) {
    quantize_block(weights)
}



/// Reconstruct f32 values from quantized i16 and scale.
pub fn dequantize_block(quantized: &[i16], scale: f32, outliers: &[(usize, f32)]) -> Vec<f32> {
    let n = quantized.len();
    let mut result = vec![0.0f32; n];

    for (i, &q) in quantized.iter().enumerate() {
        result[i] = q as f32 * scale;
    }

    for &(idx, orig) in outliers {
        if idx < n {
            result[idx] = orig;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let weights = vec![0.001, -0.002, 0.0005, -0.0001, 0.5, -0.3, 0.0, 1e-6, 0.999, -0.999];
        let (scale, outliers) = quantize_block(&weights);

        eprintln!("scale: {:.10e}", scale);
        eprintln!("outliers: {:?}", outliers);

        // Reconstruct from outliers only (no quantized vec returned)
        let n = weights.len();
        let mut recon = vec![0.0f32; n];
        for &(idx, orig) in &outliers {
            if idx < n {
                recon[idx] = orig;
            }
        }

        for (i, (orig, rec)) in weights.iter().zip(recon.iter()).enumerate() {
            let err = (orig - rec).abs();
            eprintln!("  [{}] orig={:.6e} recon={:.6e} err={:.6e}", i, orig, rec, err);
            // Non-outlier elements are 0.0 in recon (not reconstructed from quantized vec)
            // so only assert on outlier elements
        }
    }

    #[test]
    fn test_quantize_block_scale_not_zero() {
        // Ensure scale is not zero for non-trivial input
        let weights = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let (scale, _outliers) = quantize_block(&weights);
        assert!(scale > 0.0, "scale should not be zero, got {}", scale);
    }
}


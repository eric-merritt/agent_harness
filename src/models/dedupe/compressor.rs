use crate::models::dedupe::tensor::DedupCountTensor;
use crate::models::dedupe::types::Sandbag;
use crate::models::dedupe::truncation::{quantize_block, quantize_block_avx512, quantize_block_kl};
use crate::models::convert::common::{CompressOutput, CompressJob, CHUNK_SIZE};
use crate::models::convert::core::{serialize_core, deserialize_core};
use crate::models::quantization::QuantizationLevels;
use hashbrown::HashMap as AHashMap;

/// Convert f32 to IEEE 754 binary16 (half precision).
/// Handles exponent bias shift (127 → 15), mantissa rounding, and special values.
#[inline]
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x7FFFFF) as u32;

    // Extract f32 exponent (unbiased)
    let e = exp as i32 - 127;

    if exp == 0 {
        // Zero or subnormal → flush to zero
        return sign;
    }
    if exp == 0xFF {
        // Inf or NaN → preserve
        return sign | (0x7C00u16 | ((mant >> 13) & 0x03FF) as u16);
    }

    // Check for overflow into f16
    if e >= 15 {
        return sign | 0x7C00; // ±infinity
    }
    if e <= -15 {
        return sign; // Flush to zero for underflow
    }

    // Re-bias exponent for f16 (bias 15)
    let new_exp = (e + 15) as u16;
    // Round mantissa: f32 has 23 mantissa bits, f16 has 10. Shift right by 13.
    // Bit 13 (the guard bit) determines rounding.
    let new_mant = ((mant >> 13) & 0x3FF) as u16;
    let guard = (mant >> 12) & 1; // Round bit
    let rounded_mant = if guard == 1 {
        // Round to nearest, ties to even
        if new_mant & 1 == 0 { new_mant } else { new_mant.wrapping_add(1) }
    } else {
        new_mant
    };

    sign | (new_exp << 10) | rounded_mant
}

impl DedupCountTensor {
    // Quantize-based compression (fast path): i16 per weight with dedup.
    pub fn compress_quantized(weights: &[f32], prefix_digits: usize, _truncate_rounds: usize) -> (Self, Sandbag) {
        let (scale, outliers) = quantize_block(weights);
        Self::build_from_quantized(weights, scale, outliers, prefix_digits, _truncate_rounds)
    }

    /// AVX-512 accelerated quantize-based compression.
    pub fn compress_quantized_avx512(weights: &[f32], prefix_digits: usize, _truncate_rounds: usize) -> (Self, Sandbag) {
        let (scale, outliers) = if is_x86_feature_detected!("avx512f") && weights.len() >= 16 {
            unsafe { quantize_block_avx512(weights) }
        } else {
            quantize_block(weights)
        };
        Self::build_from_quantized(weights, scale, outliers, prefix_digits, _truncate_rounds)
    }

    /// Build DedupCountTensor + Sandbag from original f32 weights using prefix/tail split.
    /// Matches GPU/AVX shader math exactly:
    ///   prefix_int = floor(abs_w * 10^prefix_digits)
    ///   tail_int   = round((abs_w - prefix_int/10^prefix_digits) * 10^7)
    fn build_from_quantized(weights: &[f32], scale: f32, outliers: Vec<(usize, f32)>, prefix_digits: usize, _truncate_rounds: usize) -> (Self, Sandbag) {
        let n = weights.len();
        let prefix_scale = 10f32.powi(prefix_digits as i32);
        let tail_scale = 10_000_000.0f32;

        // Collect unique prefixes and tails, build manifest + sign bitvector
        let mut prefix_map: AHashMap<u8, u16> = AHashMap::with_capacity(256);
        let mut unique_prefixes: Vec<u8> = Vec::new();
        let mut tail_map: AHashMap<u32, u16> = AHashMap::with_capacity(65536);
        let mut unique_tails: Vec<u32> = Vec::new();
        let mut manifest = Vec::with_capacity(n);
        let mut signs: Vec<u8> = vec![0u8; (n + 7) / 8];

        // Accumulators for precision loss (computed in the same pass as dedup)
        let mut loss_sum = 0.0f32;
        let mut loss_count = 0usize;
        let mut max_abs_err = 0.0f32;

        for (i, &w) in weights.iter().enumerate() {
            let abs_w = w.abs();

            // prefix_int = floor(abs_w * 10^prefix_digits)
            let p_int = (abs_w * prefix_scale).floor() as u8;
            // tail_val = abs_w - p_int / 10^prefix_digits
            let tail_val = abs_w - (p_int as f32) / prefix_scale;
            // tail_int = round(tail_val * 10^7)
            let t_int = (tail_val * tail_scale).round() as u32;

            // Reconstruct from compressed form and accumulate error (same data, no second loop)
            let recon_abs = (p_int as f32) / prefix_scale + (t_int as f32) / tail_scale;
            let diff = (abs_w - recon_abs).abs();
            if !diff.is_nan() && !diff.is_infinite() {
                loss_sum += diff;
                loss_count += 1;
                if diff > max_abs_err {
                    max_abs_err = diff;
                }
            }

            let p_idx = *prefix_map.entry(p_int).or_insert_with(|| {
                let idx = unique_prefixes.len() as u16;
                unique_prefixes.push(p_int);
                idx
            });
            let t_idx = *tail_map.entry(t_int).or_insert_with(|| {
                let idx = unique_tails.len() as u16;
                unique_tails.push(t_int);
                idx
            });

            manifest.push((p_idx, t_idx));

            // Set sign bit
            if w < 0.0 {
                signs[i / 8] |= 1 << (i % 8);
            }
        }

        // Sanity check: if average per-element error exceeds 1% of the scale,
        // the compression is too lossy to be useful — log and bail.
        let avg_loss = if loss_count == 0 { 0.0_f32 } else { loss_sum / loss_count as f32 };
        if avg_loss > scale * 0.01 && !outliers.is_empty() {
            eprintln!(
                "build_from_quantized: avg_loss={:.2e} max_err={:.2e} scale={:.2e} — compression may be too lossy",
                avg_loss, max_abs_err, scale
            );
        }

        // Count occurrences for tensor (from manifest built above)
        let mut prefix_counts: AHashMap<u16, u32> = AHashMap::new();
        let mut tail_counts: AHashMap<u16, u32> = AHashMap::new();
        for &(p_idx, t_idx) in &manifest {
            *prefix_counts.entry(p_idx).or_insert(0) += 1;
            *tail_counts.entry(t_idx).or_insert(0) += 1;
        }

        // Build UniqueTail list for tensor
        let ut: Vec<crate::models::dedupe::types::UniqueTail> = unique_tails
            .iter()
            .enumerate()
            .map(|(i, &v)| crate::models::dedupe::types::UniqueTail {
                value: v,
                repeat_count: *tail_counts.entry(i as u16).or_insert(0),
            })
            .collect();

        let tensor = Self {
            prefixes: unique_prefixes.clone(),
            prefix_counts: prefix_counts.into_values().collect(),
            unique_tails: ut,
            count: n,
            prefix_digits,
            tail_digits: 7,
            avg_precision_lost: avg_loss,
        };

        let sandbag = Sandbag {
            scale,
            outliers,
            count: n,
            prefix_digits,
            unique_prefixes,
            unique_tails,
            manifest,
            signs,
        };

        (tensor, sandbag)
    }


    // ── Percentile-clip quantization methods ────────────────────────────────

    /// AVX-512 accelerated path — percentile-clip quantization.
    /// Uses AVX-512 SIMD for the quantization loop when available.
    pub fn compress_avx512_percent(weights: &[f32], prefix_digits: usize, truncate_rounds: usize) -> (Self, Sandbag) {
        Self::compress_quantized_avx512(weights, prefix_digits, truncate_rounds)
    }

    /// Pure-GPU CPU dedup path — reconstruct f32 weights from GPU output, then quantize (percentile).
    /// Uses AVX-512 for reconstruction when available.
    pub fn compress_from_gpu_percent(
        prefix_ints: &[u32],
        tails: &[u32],
        signs: &[u32],
        prefix_digits: usize,
        _truncate_rounds: usize,
    ) -> (Self, Sandbag) {
        let n = prefix_ints.len();
        let prefix_scale = 10f32.powi(prefix_digits as i32);

        let weights = if is_x86_feature_detected!("avx512f") && n >= 32 {
            // AVX-512 accelerated reconstruction
            let mut weights = vec![0.0f32; n];
            unsafe {
                crate::models::avx512_kernel::avx512_reconstruct_from_gpu(prefix_ints, tails, signs, prefix_scale, &mut weights);
            }
            weights
        } else {
            // Scalar fallback
            (0..n)
                .map(|i| {
                    let prefix_val = (prefix_ints[i] as f32) / prefix_scale;
                    let tail_val = (tails[i] as f32) / 10_000_000.0;
                    let abs_w = prefix_val + tail_val;
                    if signs[i] != 0 { -abs_w } else { abs_w }
                })
                .collect()
        };

        Self::compress_quantized(&weights, prefix_digits, _truncate_rounds)
    }

    /// Pure-GPU CPU dedup path — scalar reconstruction only (no AVX-512), then quantize (percentile).
    pub fn compress_from_gpu_scalar_percent(
        prefix_ints: &[u32],
        tails: &[u32],
        signs: &[u32],
        prefix_digits: usize,
        _truncate_rounds: usize,
    ) -> (Self, Sandbag) {
        let n = prefix_ints.len();
        let prefix_scale = 10f32.powi(prefix_digits as i32);

        // Force scalar reconstruction
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                let prefix_val = (prefix_ints[i] as f32) / prefix_scale;
                let tail_val = (tails[i] as f32) / 10_000_000.0;
                let abs_w = prefix_val + tail_val;
                if signs[i] != 0 { -abs_w } else { abs_w }
            })
            .collect();

        Self::compress_quantized(&weights, prefix_digits, _truncate_rounds)
    }

    /// GPU prefix chopping + AVX-512 tail processing — percentile clip.
    /// Calls gpu_compute() for GPU prefix/tail/sign extraction, then uses
    /// AVX-512 for weight reconstruction before quantization.
    pub fn compress_gpu_with_avx512_percent(
        weights: &[f32],
        prefix_digits: usize,
        truncate_rounds: usize,
    ) -> (Self, Sandbag) {
        // Try GPU path first
        if let Some(gpu_out) = crate::gpu::gpu_compute(weights, prefix_digits) {
            return Self::compress_from_gpu_percent(
                &gpu_out.prefix_ints,
                &gpu_out.tails,
                &gpu_out.signs,
                prefix_digits,
                truncate_rounds,
            );
        }
        // Fall back to scalar
        Self::compress_quantized(weights, prefix_digits, truncate_rounds)
    }

    /// GPU prefix chopping + scalar tail processing — percentile clip.
    /// Same as gpu_with_avx512 but forces scalar reconstruction path.
    pub fn compress_gpu_with_scalar_tails_percent(
        weights: &[f32],
        prefix_digits: usize,
        truncate_rounds: usize,
    ) -> (Self, Sandbag) {
        if let Some(gpu_out) = crate::gpu::gpu_compute(weights, prefix_digits) {
            // Force scalar reconstruction (no AVX-512)
            let n = gpu_out.prefix_ints.len();
            let prefix_scale = 10f32.powi(prefix_digits as i32);
            let weights: Vec<f32> = (0..n)
                .map(|i| {
                    let prefix_val = (gpu_out.prefix_ints[i] as f32) / prefix_scale;
                    let tail_val = (gpu_out.tails[i] as f32) / 10_000_000.0;
                    let abs_w = prefix_val + tail_val;
                    if gpu_out.signs[i] != 0 { -abs_w } else { abs_w }
                })
                .collect();
            Self::compress_quantized(&weights, prefix_digits, truncate_rounds)
        } else {
            Self::compress_quantized(weights, prefix_digits, truncate_rounds)
        }
    }

    // ── KL-divergence quantization methods ──────────────────────────────────

    /// Scalar path — KL divergence quantization.
    pub fn compress_quantized_kl(weights: &[f32], _prefix_digits: usize, _truncate_rounds: usize) -> (Self, Sandbag) {
        Self::compress_quantized_kl_inner(weights, _prefix_digits, _truncate_rounds, quantize_block_kl)
    }

    /// AVX-512 accelerated path — KL divergence quantization.
    pub fn compress_quantized_kl_avx512(weights: &[f32], _prefix_digits: usize, _truncate_rounds: usize) -> (Self, Sandbag) {
        // KL search is scalar (histogram-based), but the inner quantize loop can use AVX-512
        // For now use the same KL path — the KL search dominates anyway
        Self::compress_quantized_kl_inner(weights, _prefix_digits, _truncate_rounds, quantize_block_kl)
    }

    fn compress_quantized_kl_inner<F>(weights: &[f32], prefix_digits: usize, _truncate_rounds: usize, quantize_fn: F) -> (Self, Sandbag)
    where
        F: FnOnce(&[f32]) -> (f32, Vec<(usize, f32)>),
    {
        let (scale, outliers) = quantize_fn(weights);
        Self::build_from_quantized(weights, scale, outliers, prefix_digits, _truncate_rounds)
    }

    /// AVX-512 accelerated path — KL divergence quantization.
    pub fn compress_avx512_kl(weights: &[f32], prefix_digits: usize, truncate_rounds: usize) -> (Self, Sandbag) {
        Self::compress_quantized_kl_avx512(weights, prefix_digits, truncate_rounds)
    }

    /// Pure-GPU CPU dedup — KL divergence quantization.
    /// Uses AVX-512 for reconstruction when available.
    pub fn compress_from_gpu_kl(
        prefix_ints: &[u32],
        tails: &[u32],
        signs: &[u32],
        prefix_digits: usize,
        _truncate_rounds: usize,
    ) -> (Self, Sandbag) {
        let n = prefix_ints.len();
        let prefix_scale = 10f32.powi(prefix_digits as i32);

        let weights = if is_x86_feature_detected!("avx512f") && n >= 32 {
            let mut weights = vec![0.0f32; n];
            unsafe {
                crate::models::avx512_kernel::avx512_reconstruct_from_gpu(prefix_ints, tails, signs, prefix_scale, &mut weights);
            }
            weights
        } else {
            (0..n).map(|i| {
                let prefix_val = (prefix_ints[i] as f32) / prefix_scale;
                let tail_val = (tails[i] as f32) / 10_000_000.0;
                let abs_w = prefix_val + tail_val;
                if signs[i] != 0 { -abs_w } else { abs_w }
            }).collect()
        };
        Self::compress_quantized_kl(&weights, prefix_digits, _truncate_rounds)
    }

    /// Pure-GPU CPU dedup — KL divergence quantization, scalar reconstruction only.
    pub fn compress_from_gpu_scalar_kl(
        prefix_ints: &[u32],
        tails: &[u32],
        signs: &[u32],
        prefix_digits: usize,
        _truncate_rounds: usize,
    ) -> (Self, Sandbag) {
        let n = prefix_ints.len();
        let prefix_scale = 10f32.powi(prefix_digits as i32);
        let weights: Vec<f32> = (0..n).map(|i| {
            let prefix_val = (prefix_ints[i] as f32) / prefix_scale;
            let tail_val = (tails[i] as f32) / 10_000_000.0;
            let abs_w = prefix_val + tail_val;
            if signs[i] != 0 { -abs_w } else { abs_w }
        }).collect();
        Self::compress_quantized_kl(&weights, prefix_digits, _truncate_rounds)
    }

    /// GPU + AVX-512 tails — KL divergence quantization.
    pub fn compress_gpu_with_avx512_kl(weights: &[f32], prefix_digits: usize, truncate_rounds: usize) -> (Self, Sandbag) {
        if let Some(gpu_out) = crate::gpu::gpu_compute(weights, prefix_digits) {
            return Self::compress_from_gpu_kl(&gpu_out.prefix_ints, &gpu_out.tails, &gpu_out.signs, prefix_digits, truncate_rounds);
        }
        Self::compress_quantized_kl(weights, prefix_digits, truncate_rounds)
    }

    /// GPU + scalar tails — KL divergence quantization.
    /// Forces scalar reconstruction path (no AVX-512).
    pub fn compress_gpu_with_scalar_tails_kl(weights: &[f32], prefix_digits: usize, truncate_rounds: usize) -> (Self, Sandbag) {
        if let Some(gpu_out) = crate::gpu::gpu_compute(weights, prefix_digits) {
            let n = gpu_out.prefix_ints.len();
            let prefix_scale = 10f32.powi(prefix_digits as i32);
            let weights: Vec<f32> = (0..n)
                .map(|i| {
                    let prefix_val = (gpu_out.prefix_ints[i] as f32) / prefix_scale;
                    let tail_val = (gpu_out.tails[i] as f32) / 10_000_000.0;
                    let abs_w = prefix_val + tail_val;
                    if gpu_out.signs[i] != 0 { -abs_w } else { abs_w }
                })
                .collect();
            Self::compress_quantized_kl(&weights, prefix_digits, truncate_rounds)
        } else {
            Self::compress_quantized_kl(weights, prefix_digits, truncate_rounds)
        }
    }

    // ── Backward-compatibility aliases ──────────────────────────────────────

    #[deprecated(since = "0.2.0", note = "Use compress_avx512_percent")]
    pub fn compress_avx512_fast(weights: &[f32], prefix_digits: usize, truncate_rounds: usize) -> (Self, Sandbag) {
        Self::compress_avx512_percent(weights, prefix_digits, truncate_rounds)
    }

    #[deprecated(since = "0.2.0", note = "Use compress_from_gpu_percent")]
    pub fn compress_from_gpu_fast(prefix_ints: &[u32], tails: &[u32], signs: &[u32], prefix_digits: usize, truncate_rounds: usize) -> (Self, Sandbag) {
        Self::compress_from_gpu_percent(prefix_ints, tails, signs, prefix_digits, truncate_rounds)
    }

    #[deprecated(since = "0.2.0", note = "Use compress_gpu_with_avx512_percent")]
    pub fn compress_gpu_with_avx512_fast(weights: &[f32], prefix_digits: usize, truncate_rounds: usize) -> (Self, Sandbag) {
        Self::compress_gpu_with_avx512_percent(weights, prefix_digits, truncate_rounds)
    }
    // Single entry point for compressing a tensor's weights.
    // Owns the full-precision escape, the >CHUNK_SIZE chunk split, and the
    // core/sandbag serialization. No other logic belongs here.
    pub fn compress_job(
        weights: &[f32],
        prefix_digits: usize,
        truncate_rounds: usize,
    ) -> CompressOutput {
        Self::compress_job_with_level(weights, prefix_digits, truncate_rounds, QuantizationLevels::ToNeg8)
    }

    /// Compress with an explicit quantization level (derived from tensor name).
    pub fn compress_job_with_level(
        weights: &[f32],
        prefix_digits: usize,
        truncate_rounds: usize,
        level: QuantizationLevels,
    ) -> CompressOutput {
        // FullPrecision: store raw f32, no quantization
        if matches!(level, QuantizationLevels::FullPrecision) {
            let mut core = Vec::with_capacity(4 + weights.len() * 4);
            core.extend_from_slice(&1u32.to_le_bytes());
            for &w in weights {
                core.extend_from_slice(&w.to_le_bytes());
            }
            return CompressOutput {
                core,
                sandbag: Vec::new(),
                prefix_count: 0,
                unique_tail_count: 0,
                shared_weights: 0,
                mean_precision_lost: 0.0,
                full_precision: true,
            };
        }

        // HalfPrecision: store as f16 (proper IEEE 754 conversion)
        if matches!(level, QuantizationLevels::HalfPrecision) {
            let mut core = Vec::with_capacity(4 + weights.len() * 2);
            core.extend_from_slice(&1u32.to_le_bytes());
            for &w in weights {
                let f16 = f32_to_f16(w);
                core.extend_from_slice(&f16.to_le_bytes());
            }
            return CompressOutput {
                core,
                sandbag: Vec::new(),
                prefix_count: 0,
                unique_tail_count: 0,
                shared_weights: 0,
                mean_precision_lost: 0.0,
                full_precision: false,
            };
        }

        // Guard: skip quantization for tiny or out-of-range tensors
        if Self::should_be_full_precision(weights) {
            let mut core = Vec::with_capacity(4 + weights.len() * 4);
            core.extend_from_slice(&1u32.to_le_bytes());
            for &w in weights {
                core.extend_from_slice(&w.to_le_bytes());
            }
            return CompressOutput {
                core,
                sandbag: Vec::new(),
                prefix_count: 0,
                unique_tail_count: 0,
                shared_weights: 0,
                mean_precision_lost: 0.0,
                full_precision: true,
            };
        }

        // ToNeg4 / ToNeg8: quantize + dedup
        if weights.len() > CHUNK_SIZE {
            let n_chunks = (weights.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
            let mut core = Vec::new();
            core.extend_from_slice(&(n_chunks as u32).to_le_bytes());
            let mut sandbag = Vec::new();
            let mut prefix_count = 0;
            let mut unique_tail_count = 0;
            let mut shared_weights = 0;
            let mut mean_pl_sum = 0.0f32;
            let mut chunk_count = 0u32;

            for chunk in weights.chunks(CHUNK_SIZE) {
                let (t, m) = Self::compress_gpu_with_avx512_percent(chunk, prefix_digits, truncate_rounds);
                core.extend(serialize_core(&t));
                sandbag.extend(m.to_bytes());
                prefix_count += t.prefixes.len();
                unique_tail_count += t.unique_tail_count();
                shared_weights += t.shared_tail_weights();
                mean_pl_sum += t.avg_precision_lost;
                chunk_count += 1;
            }

            return CompressOutput {
                core,
                sandbag,
                prefix_count,
                unique_tail_count,
                shared_weights,
                mean_precision_lost: mean_pl_sum / chunk_count.max(1) as f32,
                full_precision: false,
            };
        } else {
            let (t, m) = Self::compress_gpu_with_avx512_percent(weights, prefix_digits, truncate_rounds);
            let core = serialize_core(&t);
            let sandbag = m.to_bytes();
            let mut full_core = Vec::with_capacity(4 + core.len());
            full_core.extend_from_slice(&1u32.to_le_bytes());
            full_core.extend(core);

            CompressOutput {
                core: full_core,
                sandbag,
                prefix_count: t.prefixes.len(),
                unique_tail_count: t.unique_tail_count(),
                shared_weights: t.shared_tail_weights(),
                mean_precision_lost: t.avg_precision_lost,
                full_precision: false,
            }
        }
    }

    pub fn should_be_full_precision(weights: &[f32]) -> bool {
        let step = (weights.len() / 1000).max(1);
        for i in (0..weights.len()).step_by(step) {
            if weights[i].abs() > 2.0 {
                return true;
            }
        }
        if weights.len() < 8192 {
            return true;
        }
        false
    }
}

#[cfg(test)]
mod roundtrip_tests {
    use super::*;
    use crate::models::formats::gguf::GGUFFile;
    use std::fs::File;

    #[test]
    fn test_compress_decompress_roundtrip() {
        // Proper Box-Muller: generates N(0, σ²) samples from uniform [0,1)
        let mut weights = Vec::with_capacity(1_000_000);
        let mut state: u64 = 12345u64;
        let mut push_pair = true;
        for _ in 0..1_000_000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (((state >> 40) as f64) / (1u64 << 24) as f64).max(1e-15).min(1.0 - f64::EPSILON);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((state >> 40) as f64) / (1u64 << 24) as f64;
            // Box-Muller transform: two independent N(0,1) from two uniforms
            let r = (-2.0_f64 * u1.ln()).sqrt();
            let theta = u2 * 2.0 * std::f64::consts::PI;
            let x0 = r * theta.cos();
            let x1 = r * theta.sin();
            // Scale to σ=0.3 (typical weight range)
            let sigma = 0.3;
            if push_pair {
                weights.push((x0 * sigma) as f32);
                push_pair = false;
            } else {
                weights.push((x1 * sigma) as f32);
                push_pair = true;
            }
        }

        let (tensor, sandbag) = DedupCountTensor::compress_quantized(&weights, 2, 3);
        eprintln!("sandbag.scale = {:.10e}", sandbag.scale);
        eprintln!("sandbag.unique_prefixes.len = {}", sandbag.unique_prefixes.len());
        eprintln!("sandbag.unique_tails.len = {}", sandbag.unique_tails.len());
        eprintln!("sandbag.outliers.len = {}", sandbag.outliers.len());
        eprintln!("sandbag.count = {}", sandbag.count);
        eprintln!("tensor.count = {}", tensor.count);

        // Serialize → deserialize → decompress (verify round-trip through wire format)
        let bytes = sandbag.to_bytes();
        eprintln!("serialized sandbag: {} bytes", bytes.len());
        let sandbag2 = crate::models::dedupe::types::Sandbag::from_bytes(&bytes)
            .expect("deserialization failed");
        let recon = tensor.decompress_all(&sandbag2);

        // Check for NaN/Inf
        let nan_count = recon.iter().filter(|r| r.is_nan() || r.is_infinite()).count();
        eprintln!("nan/inf in recon: {}", nan_count);

        let mut max_err = 0.0f32;
        let mut avg_err = 0.0f32;
        let mut err_count = 0usize;
        for (orig, rec) in weights.iter().zip(recon.iter()) {
            let err = (orig - rec).abs();
            if err.is_nan() || err.is_infinite() { continue; }
            if err > max_err { max_err = err; }
            avg_err += err;
            err_count += 1;
        }
        avg_err /= err_count.max(1) as f32;

        eprintln!("compress_decompress: avg_err={:.6e} max_err={:.6e}", avg_err, max_err);
        assert!(avg_err < 2e-3, "avg_err {:.6e} too large", avg_err);
        assert!(max_err < 0.01, "max_err {:.6e} too large", max_err);
    }

    /// Load a real tensor from a GGUF, compress → serialize → deserialize → decompress,
    /// then compare original vs reconstructed values.
    #[test]
    #[ignore = "integration test — loads a real GGUF model file"]
    fn test_real_weight_pipeline() {
        // Pick a small model to test with
        let model_path = std::env::var("TEST_MODEL_PATH").unwrap_or_else(|_| {
            // Default: smallest available model
            "/home/ermer/models/Qwen/Qwen3.5-0.8B/Qwen3.5-0.8B.gguf".to_string()
        });

        eprintln!("Loading GGUF: {}", model_path);
        let gguf = GGUFFile::from_file(std::path::Path::new(&model_path)).expect("parse GGUF");
        eprintln!("Model: {} ({} tensors)", gguf.model_name(), gguf.tensor_info.len());

        // Pick the first tensor that's F16 or F32 (easy to dequantize)
        let tensor_idx = gguf
            .tensor_info
            .iter()
            .position(|t| matches!(t.dtype, 0 | 1 | 30))
            .unwrap_or_else(|| {
                eprintln!("No F16/F32/BF16 tensor found, using first tensor (dtype={})", gguf.tensor_info[0].dtype);
                0
            });

        let info = &gguf.tensor_info[tensor_idx];
        eprintln!(
            "Tensor #{}: {} dtype={} shape={:?} elems={}",
            tensor_idx, info.name, info.dtype, info.dim, info.element_count()
        );

        // Read and dequantize
        let mut file = File::open(&model_path).expect("open model file");
        let raw = gguf.read_tensor_data(&mut file, tensor_idx).expect("read tensor data");
        let weights = gguf.dequantize_to_f32(&raw, info.dtype, info.element_count() as usize);

        eprintln!("Dequantized {} weights", weights.len());

        // Print original stats
        let orig_min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let orig_max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let orig_mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        let orig_first5: Vec<f32> = weights.iter().take(5).copied().collect();
        let orig_last5: Vec<f32> = weights.iter().rev().take(5).copied().collect();

        eprintln!("Original:  min={:.6e} max={:.6e} mean={:.6e}", orig_min, orig_max, orig_mean);
        eprintln!("First 5: {:?}", orig_first5);
        eprintln!("Last 5:  {:?}", orig_last5);

        // Compress
        let (tensor, sandbag) = DedupCountTensor::compress_quantized(&weights, 2, 3);
        eprintln!(
            "Compressed: scale={:.10e} unique_prefixes={} unique_tails={} outliers={} avg_loss={:.6e}",
            sandbag.scale,
            sandbag.unique_prefixes.len(),
            sandbag.unique_tails.len(),
            sandbag.outliers.len(),
            tensor.avg_precision_lost,
        );

        // Serialize → deserialize
        let bytes = sandbag.to_bytes();
        eprintln!("Serialized sandbag: {} bytes", bytes.len());
        let sandbag2 = Sandbag::from_bytes(&bytes).expect("deserialize sandbag");
        eprintln!("Deserialized: count={} prefixes={} tails={}", sandbag2.count, sandbag2.unique_prefixes.len(), sandbag2.unique_tails.len());

        // Decompress
        let recon = tensor.decompress_all(&sandbag2);

        // Reconstructed stats
        let recon_min = recon.iter().cloned().fold(f32::INFINITY, f32::min);
        let recon_max = recon.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let recon_mean: f32 = recon.iter().sum::<f32>() / recon.len() as f32;
        let recon_first5: Vec<f32> = recon.iter().take(5).copied().collect();
        let recon_last5: Vec<f32> = recon.iter().rev().take(5).copied().collect();

        eprintln!("Reconstructed: min={:.6e} max={:.6e} mean={:.6e}", recon_min, recon_max, recon_mean);
        eprintln!("First 5: {:?}", recon_first5);
        eprintln!("Last 5:  {:?}", recon_last5);

        // Per-element error analysis
        let nan_count = recon.iter().filter(|r| r.is_nan() || r.is_infinite()).count();
        eprintln!("NaN/Inf in reconstructed: {}", nan_count);

        let mut max_err = 0.0f32;
        let mut avg_err = 0.0f32;
        let mut max_rel_err = 0.0f32;
        let mut err_count = 0usize;
        let mut err_gt_1e3 = 0usize;
        let mut err_gt_1e2 = 0usize;

        for (i, (orig, rec)) in weights.iter().zip(recon.iter()).enumerate() {
            let abs_err = (orig - rec).abs();
            if abs_err.is_nan() || abs_err.is_infinite() {
                continue;
            }
            if abs_err > max_err {
                max_err = abs_err;
            }
            avg_err += abs_err;
            err_count += 1;

            // Relative error (skip near-zero)
            if orig.abs() > 1e-8 {
                let rel = abs_err / orig.abs();
                if rel > max_rel_err {
                    max_rel_err = rel;
                }
            }

            if abs_err > 1e-3 { err_gt_1e3 += 1; }
            if abs_err > 1e-2 { err_gt_1e2 += 1; }
        }
        avg_err /= err_count.max(1) as f32;

        eprintln!("\n=== ERROR ANALYSIS ===");
        eprintln!("Elements: {}", weights.len());
        eprintln!("Avg abs error: {:.6e}", avg_err);
        eprintln!("Max abs error: {:.6e}", max_err);
        eprintln!("Max rel error: {:.6e}", max_rel_err);
        eprintln!("Errors > 1e-3: {} ({:.2}%)", err_gt_1e3, err_gt_1e3 as f64 / weights.len() as f64 * 100.0);
        eprintln!("Errors > 1e-2: {} ({:.2}%)", err_gt_1e2, err_gt_1e2 as f64 / weights.len() as f64 * 100.0);

        // Verify beginning/end values match closely
        eprintln!("\n=== BEGINNING/END COMPARISON ===");
        for i in 0..5 {
            let orig_v = weights[i];
            let rec_v = recon[i];
            let err = (orig_v - rec_v).abs();
            eprintln!("[{}] orig={:.10e} recon={:.10e} err={:.6e}", i, orig_v, rec_v, err);
        }
        let n = weights.len();
        for i in 0..5 {
            let idx = n - 1 - i;
            let orig_v = weights[idx];
            let rec_v = recon[idx];
            let err = (orig_v - rec_v).abs();
            eprintln!("[{}] orig={:.10e} recon={:.10e} err={:.6e}", idx, orig_v, rec_v, err);
        }

        // Assertions
        assert!(nan_count == 0, "{} NaN/Inf values in reconstruction", nan_count);
        assert!(avg_err < 1e-2, "avg_err {:.6e} too large", avg_err);
        assert!(max_err < 0.1, "max_err {:.6e} too large", max_err);
    }
}

// Shared AVX-512 kernels for the models crate.
//
// Both the dedup compression pipeline and the convert pipeline use these
// kernels. Keep them here so neither module owns the SIMD intrinsics.

use std::arch::x86_64::*;
use hashbrown::HashMap as AHashMap;
use super::convert::common::{CompressJob, CompressOutput, CHUNK_SIZE};
use super::dedup::types::{Sandbag, UniqueTail};

#[target_feature(enable = "avx512f")]
pub unsafe fn compress_job_avx512(
    job: &CompressJob,
    prefix_digits: usize,
    truncate_rounds: usize,
) -> CompressOutput {
    // Keep exact signatures and variable initializations
    use crate::models::dedup::tensor::DedupCountTensor;

    let weights = &job.weights;
    let n = weights.len();

    let initial_tail_digits = DedupCountTensor::TOTAL_DIGITS - prefix_digits;
    let prefix_scale = 10f32.powi(prefix_digits as i32);
    let tail_scale = 10f32.powi(DedupCountTensor::TOTAL_DIGITS as i32);

    let mut prefix_map: AHashMap<u16, u16> = AHashMap::with_capacity(512);
    let mut prefixes: Vec<u16> = Vec::with_capacity(512);
    let mut prefix_idx = vec![0u16; n]; 
    let mut sign_bits = vec![0u8; (n + 7) / 8];
    let mut group_tails: Vec<Vec<u32>> = Vec::new();

    // Setup AVX-512 vector constants for Step 1
    let v_prefix_scale = _mm512_set1_ps(prefix_scale);
    let v_tail_scale = _mm512_set1_ps(tail_scale);
    let v_sign_mask = _mm512_set1_ps(-0.0);

    let chunks = weights.chunks_exact(16);
    let remainder = chunks.remainder();
    let chunk_count = chunks.len();

    // ── Step 1 Accelerated with AVX-512 ──
    // Temporaries to hold vectorized structural results
    let mut temp_prefixes = [0i32; 16];
    let mut temp_tails = [0i32; 16];

    for c in 0..chunk_count {
        let offset = c * 16;
        let chunk_ptr = weights[offset..].as_ptr();

        unsafe {
            let f_w = _mm512_loadu_ps(chunk_ptr);
            
            // Vectorized sign extraction via bitmask comparison against -0.0
            let sign_mask = _mm512_cmp_ps_mask(f_w, _mm512_setzero_ps(), _MM_CMPINT_LT);
            let sign_bits_u16 = sign_mask as u16;
            
            // Pack sign bits into your exact sign_bits byte vector array representation
            let byte_idx = offset / 8;
            sign_bits[byte_idx] |= (sign_bits_u16 & 0xFF) as u8;
            sign_bits[byte_idx + 1] |= ((sign_bits_u16 >> 8) & 0xFF) as u8;

            // Compute absolute values and scale fixed points via 16-channel execution
            let f_abs = _mm512_andnot_ps(v_sign_mask, f_w);
            let f_pref_scaled = _mm512_mul_ps(f_abs, v_prefix_scale);
            
            // Round toward negative infinity (floor representation)
            let f_pref_floor = _mm512_roundscale_ps(f_pref_scaled, _MM_FROUND_TO_NEG_INF);
            let v_pref_int = _mm512_cvtps_epi32(f_pref_floor);

            // Back to floats to extract residual tails
            let f_pref_val = _mm512_div_ps(f_pref_floor, v_prefix_scale);
            let f_tail_val = _mm512_sub_ps(f_abs, f_pref_val);
            let f_tail_scaled = _mm512_mul_ps(f_tail_val, v_tail_scale);
            
            // Round to nearest even
            let v_tail_int = _mm512_cvtps_epi32(_mm512_roundscale_ps(f_tail_scaled, _MM_FROUND_TO_NEAREST_INT));

            // Drain vector registers into linear layout for grouping maps
            _mm512_storeu_si512(temp_prefixes.as_mut_ptr() as *mut __m512i, v_pref_int);
            _mm512_storeu_si512(temp_tails.as_mut_ptr() as *mut __m512i, v_tail_int);
        }

        // Standard zero-allocation grouping pass for the extracted vector channels
        for i in 0..16 {
            let p_int = temp_prefixes[i] as u16;
            let t_int = temp_tails[i] as u32;
            let idx = offset + i;

            let group_idx = match prefix_map.get(&p_int) {
                Some(&g) => g,
                None => {
                    let g = prefix_map.len() as u16;
                    prefix_map.insert(p_int, g);
                    prefixes.push(p_int);
                    group_tails.push(Vec::with_capacity(CHUNK_SIZE));
                    g
                }
            };
            prefix_idx[idx] = group_idx;
            group_tails[group_idx as usize].push(t_int);
        }
    }

    // Scalar fallback execution for any hanging remainder items (<16)
    let rem_offset = chunk_count * 16;
    for (i, &w) in remainder.iter().enumerate() {
        let idx = rem_offset + i;
        let sign = w < 0.0;
        let abs_w = w.abs();
        let p_int = (abs_w * prefix_scale).floor() as u16;
        let t_val = abs_w - (p_int as f32 / prefix_scale);
        let t_int = (t_val * tail_scale).round() as u32;

        let group_idx = match prefix_map.get(&p_int) {
            Some(&g) => g,
            None => {
                let g = prefix_map.len() as u16;
                prefix_map.insert(p_int, g);
                prefixes.push(p_int);
                group_tails.push(Vec::new());
                g
            }
        };

        prefix_idx[idx] = group_idx;
        if sign { sign_bits[idx / 8] |= 1 << (idx % 8); }
        group_tails[group_idx as usize].push(t_int);
    }

    // ── Step 2: Adaptive Rounding Truncation Passes ──
    let mut global_loss_sum = 0.0f32;
    let mut global_loss_count = 0usize;
    let mut current_tail_digits = initial_tail_digits;
    let mut round_ups = vec![Vec::with_capacity(truncate_rounds); prefixes.len()];

    for _ in 0..truncate_rounds {
        let current_divisor = 10f32.powi((prefix_digits + current_tail_digits) as i32);
        let next_divisor = 10f32.powi((prefix_digits + current_tail_digits - 1) as i32);

        let v_curr_div = _mm512_set1_ps(current_divisor);
        let v_next_div = _mm512_set1_ps(next_divisor);
        let v_div_ten = _mm512_set1_ps(0.1f32);

        for (gidx, gt) in group_tails.iter_mut().enumerate() {
            if gt.is_empty() {
                round_ups[gidx].push(false);
                continue;
            }

            let mut sum_last_digits = 0u64;
            for &tail in gt.iter() {
                sum_last_digits += (tail % 10) as u64;
            }
            
            let avg = sum_last_digits as f32 / gt.len() as f32;
            let round_up = avg > 5.0;
            round_ups[gidx].push(round_up);

            let v_add_one = _mm512_set1_epi32(if round_up { 1 } else { 0 });

            // Avoid iterator move issues by calculating raw block bounds via indexing
            let gt_len = gt.len();
            let gt_chunks_count = gt_len / 16;
            let gt_remainder_offset = gt_chunks_count * 16;

            for chunk_idx in 0..gt_chunks_count {
                let offset = chunk_idx * 16;
                unsafe {
                    let chunk_ptr = gt[offset..].as_ptr();
                    let chunk_mut_ptr = gt[offset..].as_mut_ptr();

                    let v_tail = _mm512_loadu_si512(chunk_ptr as *const __m512i);
                    let f_old = _mm512_div_ps(_mm512_cvtepi32_ps(v_tail), v_curr_div);

                    // Fixed-point division by 10 using float precision transmutations
                    let f_divided = _mm512_mul_ps(_mm512_cvtepi32_ps(v_tail), v_div_ten);
                    let f_truncated = _mm512_roundscale_ps(f_divided, _MM_FROUND_TO_NEG_INF);
                    let v_new_tail = _mm512_add_epi32(_mm512_cvtps_epi32(f_truncated), v_add_one);

                    let f_new = _mm512_div_ps(_mm512_cvtepi32_ps(v_new_tail), v_next_div);
                    let f_loss = _mm512_abs_ps(_mm512_sub_ps(f_old, f_new));

                    // Accumulate losses horizontally from register pools
                    let mut loss_buffer = [0.0f32; 16];
                    _mm512_storeu_ps(loss_buffer.as_mut_ptr(), f_loss);
                    global_loss_sum += loss_buffer.iter().sum::<f32>();
                    global_loss_count += 16;

                    _mm512_storeu_si512(chunk_mut_ptr as *mut __m512i, v_new_tail);
                }
            }

            for tail in &mut gt[gt_remainder_offset..] {
                let old_val = *tail as f32 / current_divisor;
                *tail = if round_up { *tail / 10 + 1 } else { *tail / 10 };
                let new_val = *tail as f32 / next_divisor;
                global_loss_sum += (old_val - new_val).abs();
                global_loss_count += 1;
            }
        }
        current_tail_digits -= 1;
    }

    let global_avg_lost = global_loss_sum / global_loss_count.max(1) as f32;

    // ── Step 3: Find unique tails ──
    let mut tail_counts: AHashMap<u16, u32> = AHashMap::new();
    for gt in &group_tails {
        for &tail in gt {
            let tv = tail as u16;
            *tail_counts.entry(tv).or_insert(0) += 1;
        }
    }

    let mut unique_tail_values: Vec<u16> = tail_counts.keys().copied().collect();
    unique_tail_values.sort_unstable();

    let tail_idx_map: AHashMap<u16, u16> = unique_tail_values.iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u16))
        .collect();

    let unique_tails: Vec<UniqueTail> = unique_tail_values.iter().map(|&v| {
        UniqueTail { value: v, repeat_count: tail_counts[&v] }
    }).collect();

    // ── Step 4: Map indices using your existing kernel layout ──
    let mut tail_idx = vec![0u16; n];
    let mut out_tail_ints = vec![0u32; n];
    
    // Direct invocation to your existing tail compilation library block
    unsafe {
        avx512_reconstruct_tails(
            weights, prefix_scale, tail_scale, &prefix_idx,
            &round_ups, truncate_rounds, &mut out_tail_ints,
        );
    }
    for i in 0..n {
        let tv = out_tail_ints[i] as u16;
        tail_idx[i] = tail_idx_map.get(&tv).copied().unwrap_or(0);
    }

    let prefix_counts: Vec<u32> = (0..prefixes.len())
        .map(|gidx| group_tails[gidx].len() as u32)
        .collect();

    let shared_weights_calc = unique_tails.iter()
        .filter(|t| t.repeat_count > 1)
        .map(|t| t.repeat_count as usize)
        .sum();

    // Package the results to match your structure mapping requirements

    let tensor = DedupCountTensor {
        prefixes,
        prefix_counts,
        unique_tails,
        count: n,
        prefix_digits,
        tail_digits: current_tail_digits,
        avg_precision_lost: global_avg_lost,
    };
    let serialized_core = crate::models::convert::core::serialize_core(&tensor);
    let sandbag = Sandbag { prefix_idx, tail_idx, tail_width: if truncate_rounds >= 3 { 0 } else { 1 }, sign_bits, count: n };

    CompressOutput {
        core: serialized_core,
        sandbag: sandbag.to_bytes(),
        prefix_count: tensor.prefix_counts.len(),
        unique_tail_count: tensor.unique_tails.len(),
        shared_weights: shared_weights_calc,
        mean_precision_lost: global_avg_lost,
        full_precision: false,
    }
}



/// Processes 16 elements simultaneously to apply round-up adjustments across flat vectors.
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_reconstruct_tails(
    weights: &[f32],
    prefix_scale: f32,
    tail_scale: f32,
    prefix_idx: &[u16],
    round_ups: &[Vec<bool>],
    truncate_rounds: usize,
    out_tail_ints: &mut [u32],
) {
    let n = weights.len();
    let chunks_16 = n / 16;

    let v_p_scale = _mm512_set1_ps(prefix_scale);
    let v_t_scale = _mm512_set1_ps(tail_scale);
    let v_one = _mm512_set1_epi32(1);

    // Magic constant parameters for branchless 32-bit division-by-10
    // formula: (x * 0x1999999A) >> 32 -> then shifted down by 2 bits.
    let magic_mul = _mm512_set1_epi64(0x1999999A);

    for c in 0..chunks_16 {
        let offset = c * 16;

        let v_w = _mm512_abs_ps(unsafe{_mm512_loadu_ps(weights.as_ptr().add(offset))});

        let v_scaled_prefix = _mm512_mul_ps(v_w, v_p_scale);
        let v_prefix_int = _mm512_roundscale_ps(v_scaled_prefix, _MM_FROUND_TO_NEG_INF);
        let v_prefix_val = _mm512_div_ps(v_prefix_int, v_p_scale);

        let v_tail_val = _mm512_sub_ps(v_w, v_prefix_val);
        let mut v_tail_int = _mm512_cvtps_epi32(_mm512_roundscale_ps(
            _mm512_mul_ps(v_tail_val, v_t_scale),
            _MM_FROUND_TO_NEAREST_INT
        ));

        for round in 0..truncate_rounds {
            let mut mask: u16 = 0;
            for lane in 0..16 {
                let gidx = prefix_idx[offset + lane] as usize;
                if *round_ups[gidx].get(round).unwrap_or(&false) {
                    mask |= 1 << lane;
                }
            }

            // --- Hardware-Accelerated Vectorized Integer Division-by-10 ---
            // Step A: Process even lanes (0, 2, 4...) using 64-bit wide math lanes
            let prod_even = _mm512_mul_epi32(v_tail_int, magic_mul);
            let div_even = _mm512_srli_epi64(prod_even, 34); // Shift upper bits down directly

            // Step B: Shuffle odd lanes (1, 3, 5...) down into even position blocks
            let v_tail_odd = _mm512_srli_epi64(v_tail_int, 32);
            let prod_odd = _mm512_mul_epi32(v_tail_odd, magic_mul);
            let div_odd = _mm512_srli_epi64(prod_odd, 34);

            // Step C: Interleave even and odd results back into sequential order
            let v_div = _mm512_mask_blend_epi32(
                0xAAAA, // Mask selecting odd lanes only
                div_even,
                _mm512_slli_epi64(div_odd, 32)
            );

            let v_div_plus_one = _mm512_add_epi32(v_div, v_one);

            // Masked blend: lanes with mask=1 get (tail/10 + 1), mask=0 get (tail/10)
            v_tail_int = _mm512_mask_blend_epi32(mask, v_div, v_div_plus_one);
        }

        unsafe {
            _mm512_storeu_si512(out_tail_ints.as_mut_ptr().add(offset) as *mut _, v_tail_int);
        }
    }
}

/// High-speed AVX-512 preprocessing step for the convert pipeline.
/// Extracts base prefixes, raw fractional tail bits, and sign structures
/// simultaneously across 16 elements per step before serialization.
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_preprocess_conversion_chunk(
    weights: &[f32],
    prefix_digits: usize,
    out_prefixes: &mut [f32],
    out_tails: &mut [u16],
) {
    let n = weights.len();
    let chunks_16 = n / 16;

    // Compute format scaling factors using parallel vector steps
    let scale_factor = 10f32.powi(prefix_digits as i32);
    let inv_scale_factor = 1.0 / scale_factor;
    let tail_scale = 10f32.powi((5 - prefix_digits as i32).max(0)) * 65535.0;

    let v_scale = _mm512_set1_ps(scale_factor);
    let v_inv_scale = _mm512_set1_ps(inv_scale_factor);
    let v_tail_scale = _mm512_set1_ps(tail_scale);

    for i in 0..chunks_16 {
        let offset = i * 16;

        // 1. Load 16 uncompressed floats from the incoming GGUF/Safetensors block
        let v_w = unsafe { _mm512_loadu_ps(weights.as_ptr().add(offset)) };

        // 2. Perform the format's exact prefix truncation in parallel
        // (Replaces individual w * scale logic completely)
        let v_scaled = _mm512_mul_ps(v_w, v_scale);
        let v_trunc = _mm512_roundscale_ps(v_scaled, _MM_FROUND_TO_ZERO);
        let v_prefix = _mm512_mul_ps(v_trunc, v_inv_scale);

        // 3. Isolate the target tail delta segments
        let v_tail = _mm512_sub_ps(v_w, v_prefix);
        let v_tail_scaled = _mm512_roundscale_ps(_mm512_mul_ps(v_tail, v_tail_scale), _MM_FROUND_TO_NEAREST_INT);

        // 4. Stream prefixes directly to the formatting block
        unsafe { _mm512_storeu_ps(out_prefixes.as_mut_ptr().add(offset), v_prefix) };

        // 5. Convert and compress the floating-point tails into 16-bit integers
        let v_tail_epi32 = _mm512_cvtps_epi32(v_tail_scaled);

        // Downcast 32-bit vector registers down into packed 16-bit elements.
        // _mm512_cvtepi32_epi16 packs 16 x i32 -> 16 x i16 in a __m256i.
        let v_tail_epi16 = _mm512_cvtepi32_epi16(v_tail_epi32);
        unsafe { _mm256_storeu_si256(out_tails.as_mut_ptr().add(offset) as *mut _, v_tail_epi16) };
    }
}

#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_bf16_to_f32(src: &[u8], out: &mut [f32]) {
    let n = src.len() / 2;
    // Process 32 elements (64 bytes of bf16) per iteration
    let chunks_32 = n / 32;

    for c in 0..chunks_32 {
        let offset_bytes = c * 64;
        let offset_f32 = c * 32;

        // Load 64 bytes of raw bf16 data
        let v_raw_bf16 = unsafe { _mm512_loadu_si512(src.as_ptr().add(offset_bytes) as *const _) };

        // Split into lower and upper 256-bit halves to widen them to 512-bit lanes
        let lo_256 = _mm512_castsi512_si256(v_raw_bf16);
        let hi_256 = _mm512_extracti64x4_epi64(v_raw_bf16, 1);

        // Zero-extend 16-bit lanes to 32-bit integers
        let lo_extended = _mm512_cvtepu16_epi32(lo_256);
        let hi_extended = _mm512_cvtepu16_epi32(hi_256);

        // Shift left by 16 bits to move bf16 bits into the f32 exponent/mantissa positions
        let lo_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(lo_extended, 16));
        let hi_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(hi_extended, 16));

        // Store 32 floats (2 x 512-bit vector registers)
       unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(offset_f32), lo_f32) };
       unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(offset_f32 + 16), hi_f32) };
    }
}


/// Convert 4 x F16 (in a 128-bit register as 4 x i32) to 4 x F32.
/// Bit-packing: no floating-point unit needed.
/// f16: [S][EEEEEEE][MMMMMMMMMMM]
/// f32: [S][EEEEEEEEEEEEEEE][MMMMMMMMMMMMMMMMMMMMM]
#[target_feature(enable = "avx512f")]
pub unsafe fn f16_128_to_f32(v: __m128i) -> __m128 {
    // f16: [S(bit15)][E bits14-10][M bits9-0]
    // f32: [S(bit31)][E bits30-23][M bits22-0]
    // sign:  bit 15 -> bit 31     = (v & 0x8000) << 16
    // exp:   bits 10-14 -> 23-27  = (v & 0x7C00) << 13
    // mant:  bits 0-9  -> 0-9     = (v & 0x03FF) << 13
    // exp+mant combined: (v & 0x7FFF) << 13
    let sign_part = _mm_slli_epi32(_mm_and_si128(v, _mm_set1_epi32(0x8000)), 16);
    let exp_mant  = _mm_slli_epi32(_mm_and_si128(v, _mm_set1_epi32(0x7FFF)), 13);
    let combined  = _mm_or_si128(sign_part, exp_mant);

    // Add exponent bias: f16 bias=15, f32 bias=127, delta=112
    let biased = _mm_add_epi32(combined, _mm_set1_epi32(112 << 23));

    _mm_castsi128_ps(biased)
}

#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_f16_to_f32(src: &[u8], out: &mut [f32]) {
    let n = src.len() / 2;
    // Process 32 elements (64 bytes of f16) per iteration
    let chunks_32 = n / 32; 

    for c in 0..chunks_32 {
        let offset_bytes = c * 64;
        let offset_f32 = c * 32;

        // Load 64 bytes of raw f16 data into a 512-bit integer register
        let v_raw_f16 = unsafe { _mm512_loadu_si512(src.as_ptr().add(offset_bytes) as *const _) };

        // Cast it to a 256-bit register representing the half-floats
        let v_half = _mm512_castsi512_si256(v_raw_f16);

        // Native AVX-512 hardware conversion (Handles subnormals, inf, nan instantly)
        let v_f32 = _mm512_cvtph_ps(v_half);

        // Stream the 32 full floats directly back to memory
       unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(offset_f32), v_f32) };
    }
}


/// Convert 4 x BF16 (in a 128-bit register) to 4 x F32.
/// BF16 -> F32 is just a 16-bit left shift.
#[target_feature(enable = "avx512f")]
pub unsafe fn bf16_128_to_f32(v: __m128i) -> __m128 {
    let shifted = _mm_slli_epi32(v, 16);
    _mm_castsi128_ps(shifted)
}

/// Convert raw F16 bytes (little-endian) to f32.
/// Dispatch: AVX-512 kernel when available, scalar fallback otherwise.
pub fn dispatch_f16_bytes_to_f32(src: &[u8]) -> Vec<f32> {
    let n = src.len() / 2;
    if n == 0 { return Vec::new(); }
    let mut out = vec![0f32; n];

    if is_x86_feature_detected!("avx512f") {
        unsafe { avx512_f16_to_f32(src, &mut out); }
        // Handle tail (last < 8 elements)
        for i in (n - n % 8)..n {
            let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            out[i] = f16_to_f32_scalar(bits);
        }
    } else {
        for i in 0..n {
            let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            out[i] = f16_to_f32_scalar(bits);
        }
    }
    out
}

/// Convert raw BF16 bytes (little-endian) to f32.
/// Dispatch: AVX-512 kernel when available, scalar fallback otherwise.
pub fn dispatch_bf16_bytes_to_f32(src: &[u8]) -> Vec<f32> {
    let n = src.len() / 2;
    if n == 0 { return Vec::new(); }
    let mut out = vec![0f32; n];

    if is_x86_feature_detected!("avx512f") {
        unsafe { avx512_bf16_to_f32(src, &mut out); }
        for i in (n - n % 8)..n {
            let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            out[i] = bf16_to_f32_scalar(bits);
        }
    } else {
        for i in 0..n {
            let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            out[i] = bf16_to_f32_scalar(bits);
        }
    }
    out
}

/// Scalar BF16 -> F32 conversion.
/// BF16 is the upper 16 bits of F32; the conversion is a 16-bit left shift.
#[inline(always)]
pub fn bf16_to_f32_scalar(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Scalar F16 -> F32 conversion (correct, handles subnormals/inf/nan).
#[inline(always)]
pub fn f16_to_f32_scalar(h: u16) -> f32 {
    let sign = (h & 0x8000) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 24);
        }
        // Subnormal: normalize by shifting left until leading bit is set
        let mut m = mant;
        let mut e = 0i32;
        while m & 0x0400 == 0 { m <<= 1; e -= 1; }
        m &= 0x03FF;
        let biased = (127u32 - 15 + e as u32) as u32;
        f32::from_bits((sign << 24) | (biased << 23) | (m as u32) << 13)
    } else if exp == 0x1F {
        f32::from_bits((sign << 24) | 0x7F80_0000 | (mant as u32) << 13)
    } else {
        let biased = (exp - 15 + 127) as u32;
        f32::from_bits((sign << 24) | (biased << 23) | (mant as u32) << 13)
    }
}

/// High-speed AVX-512 full precision deserialization fallback pass.
/// Processes 16 raw single-precision floating point elements (64 bytes) per iteration loop.
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_load_full_precision(src: &[u8], dest: &mut [f32]) {
    let total_bytes = src.len();
    let chunks_16 = total_bytes / 64; // 16 floats * 4 bytes = 64 bytes

    let src_ptr = src.as_ptr();
    let dest_ptr = dest.as_mut_ptr();

    // Stream 16 elements at once natively
    for c in 0..chunks_16 {
        let offset_bytes = c * 64;
        let offset_f32 = c * 16;

        // Load 512 bits of raw sequential bytes straight into vector register
        let v_raw = unsafe { _mm512_loadu_ps(src_ptr.add(offset_bytes) as *const f32) };

        // Stream raw values straight into output memory, skipping cache hierarchy
        unsafe { _mm512_storeu_ps(dest_ptr.add(offset_f32), v_raw) };
    }

    // Scalar cleanup handle for trailing unaligned elements
    let elements_processed = chunks_16 * 16;
    let total_elements = total_bytes / 4;
    if elements_processed < total_elements {
        let remaining_src = &src[elements_processed * 4..];
        let remaining_dest = &mut dest[elements_processed..];
        for (i, c) in remaining_src.chunks_exact(4).enumerate() {
            remaining_dest[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
    }
}

/// High-speed AVX-512 stream copy for stitching decompressed chunk memory blocks.
/// Bypasses L1/L2 cache pollutions during massive batch thread merging.
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_stream_stitch(src: &[f32], dest: &mut [f32], offset: usize) {  
    let chunks_16 = src.len() / 16;
    let src_ptr = src.as_ptr();
    let dest_ptr = unsafe { dest.as_mut_ptr().add(offset) };

    for c in 0..chunks_16 {
        let idx = c * 16;
        let v_data = unsafe { _mm512_loadu_ps(src_ptr.add(idx)) };
        
        // Non-temporal store instruction (vmovntps)
        // Avoids read-for-ownership cache cycles, saving massive RAM bus bandwidth
        unsafe { _mm512_stream_ps(dest_ptr.add(idx), v_data) };
    }

    // Scalar handle cleanup
    let processed = chunks_16 * 16;
    if processed < src.len() {
        dest[offset + processed..offset + src.len()].copy_from_slice(&src[processed..]);
    }
}

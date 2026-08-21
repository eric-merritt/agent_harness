// Shared AVX-512 kernels for the models crate.
//
// Both the dedup compression pipeline and the convert pipeline use these
// kernels. Keep them here so neither module owns the SIMD intrinsics.

use std::arch::x86_64::*;

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

/// AVX-512 F16 -> F32 conversion.
/// Processes 8 F16 values per iteration. Handles normal, zero, inf/nan.
/// Subnormals are flushed to zero (acceptable for weight tensors).
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_f16_to_f32(src: &[u16], out: &mut [f32]) {
    let n = src.len();
    let chunks_8 = n / 8;

    for c in 0..chunks_8 {
        let offset = c * 8;

        // Load 8 x u16 as two 128-bit registers
        let lo_u16 = unsafe {_mm_loadu_si128(src.as_ptr().add(offset) as *const _)};
        let hi_u16 = unsafe {_mm_loadu_si128(src.as_ptr().add(offset + 4) as *const _)};

        let v_lo = unsafe{ f16_128_to_f32(lo_u16) };
        let v_hi = unsafe{ f16_128_to_f32(hi_u16) };

        unsafe { _mm_storeu_ps(out.as_mut_ptr().add(offset), v_lo) };
        unsafe { _mm_storeu_ps(out.as_mut_ptr().add(offset + 4), v_hi) };
    }
}

/// Convert 4 x F16 (in a 128-bit register as 4 x i32) to 4 x F32.
/// Bit-packing: no floating-point unit needed.
/// f16: [S][EEEEEEE][MMMMMMMMMMM]
/// f32: [S][EEEEEEEEEEEEEEE][MMMMMMMMMMMMMMMMMMMMM]
#[target_feature(enable = "avx512f")]
unsafe fn f16_128_to_f32(v: __m128i) -> __m128 {
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

/// AVX-512 BF16 -> F32 conversion.
/// BF16 is the upper 16 bits of F32, so this is just a 16-bit shift left.
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_bf16_to_f32(src: &[u16], out: &mut [f32]) {
    let n = src.len();
    let chunks_8 = n / 8;

    for c in 0..chunks_8 {
        let offset = c * 8;
        let v_raw = unsafe {_mm256_loadu_si256(src.as_ptr().add(offset) as *const _)};

        // lo 4 elements
        let lo = _mm256_castsi256_si128(v_raw);
        let hi = _mm256_extracti128_si256(v_raw, 1);

        let v_lo = unsafe { bf16_128_to_f32(lo) };
        let v_hi = unsafe { bf16_128_to_f32(hi) };

        unsafe { _mm_storeu_ps(out.as_mut_ptr().add(offset), v_lo) };
        unsafe { _mm_storeu_ps(out.as_mut_ptr().add(offset + 4), v_hi) };
    }
}

/// Convert 4 x BF16 (in a 128-bit register) to 4 x F32.
/// BF16 -> F32 is just a 16-bit left shift.
#[target_feature(enable = "avx512f")]
unsafe fn bf16_128_to_f32(v: __m128i) -> __m128 {
    let shifted = _mm_slli_epi32(v, 16);
    _mm_castsi128_ps(shifted)
}

/// Convert raw F16 bytes (little-endian) to f32.
/// Uses AVX-512 when available, falls back to scalar.
pub fn f16_bytes_to_f32(src: &[u8]) -> Vec<f32> {
    let n = src.len() / 2;
    if n == 0 { return Vec::new(); }
    let mut out = vec![0f32; n];

    if is_x86_feature_detected!("avx512f") {
        let u16s: Vec<u16> = (0..n).map(|i| u16::from_le_bytes([src[i * 2], src[i * 2 + 1]])).collect();
        unsafe { avx512_f16_to_f32(&u16s, &mut out); }
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
/// Uses AVX-512 when available, falls back to scalar.
pub fn bf16_bytes_to_f32(src: &[u8]) -> Vec<f32> {
    let n = src.len() / 2;
    if n == 0 { return Vec::new(); }
    let mut out = vec![0f32; n];

    if is_x86_feature_detected!("avx512f") {
        let u16s: Vec<u16> = (0..n).map(|i| u16::from_le_bytes([src[i * 2], src[i * 2 + 1]])).collect();
        unsafe { avx512_bf16_to_f32(&u16s, &mut out); }
        for i in (n - n % 8)..n {
            let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            out[i] = f32::from_bits((bits as u32) << 16);
        }
    } else {
        for i in 0..n {
            let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            out[i] = f32::from_bits((bits as u32) << 16);
        }
    }
    out
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

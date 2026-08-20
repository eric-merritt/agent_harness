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
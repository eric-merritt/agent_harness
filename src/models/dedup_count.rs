use std::arch::x86_64::*;
use hashbrown::HashMap as AHashMap;

// ============================================================================
// ── SECTION 1: SHARED CONVERSION DATA STRUCTURES & FLAGS ───────────────────
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub enum DataFlag {
    GapFlag = 0xFD,
    TailFlag = 0xFE,
    CountFlag = 0xFF,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UniqueTail {
    pub value: u16,
    pub repeat_count: u32,
}

#[derive(Clone, Debug)]
pub struct Sandbag {
    pub prefix_idx: Vec<u16>, 
    pub tail_idx: Vec<u16>,
    pub tail_width: u8,
    pub sign_bits: Vec<u8>,
    pub count: usize,
}

#[derive(Clone, Debug)]
pub struct GlobalTable {
    pub prefix_digits: usize,
    pub prefixes: Vec<f32>,
    pub tails_for_prefix: Vec<Vec<u32>>,
}

// ============================================================================
// ── SECTION 2: DEDUP COUNT TENSOR IMPLEMENTATION ────────────────────────────
// ============================================================================

#[derive(Clone, Debug)]
pub struct DedupCountTensor {
    pub prefixes: Vec<u16>,      
    pub prefix_counts: Vec<u32>,
    pub unique_tails: Vec<UniqueTail>,
    pub count: usize,
    pub prefix_digits: usize,
    pub tail_digits: usize,
    pub avg_precision_lost: f32,
}

impl DedupCountTensor {
    const TOTAL_DIGITS: usize = 7;

    pub fn unique_tail_count(&self) -> usize {
        self.unique_tails.len()
    }

    pub fn shared_tail_weights(&self) -> usize {
        self.unique_tails.iter()
            .filter(|ut| ut.repeat_count > 1)
            .map(|ut| ut.repeat_count as usize)
            .sum()
    }

    pub fn compress(weights: &[f32], prefix_digits: usize, truncate_rounds: usize) -> (Self, Sandbag) {
        let initial_tail_digits = Self::TOTAL_DIGITS - prefix_digits;
        let prefix_scale = 10f32.powi(prefix_digits as i32);
        let tail_scale = 10f32.powi(Self::TOTAL_DIGITS as i32);
        let n = weights.len();

        // ── Step 1: Group by prefix ──
        let mut prefix_map: AHashMap<u16, u16> = AHashMap::with_capacity(512);
        let mut prefixes: Vec<u16> = Vec::new();
        let mut prefix_idx = vec![0u16; n]; 
        let mut sign_bits = vec![0u8; (n + 7) / 8];
        let mut group_tails: Vec<Vec<u32>> = Vec::new();

        for (i, &w) in weights.iter().enumerate() {
            let sign = w < 0.0;
            let abs_w = w.abs();
            let prefix_int = (abs_w * prefix_scale).floor() as u16;
            let prefix_val = prefix_int as f32 / prefix_scale;
            let tail_val = abs_w - prefix_val;
            let tail_int = (tail_val * tail_scale).round() as u32;

            let group_idx = match prefix_map.get(&prefix_int) {
                Some(&idx) => idx,
                None => {
                    let idx = prefix_map.len() as u16;
                    prefix_map.insert(prefix_int, idx);
                    prefixes.push(prefix_int);
                    group_tails.push(Vec::new());
                    idx
                }
            };

            prefix_idx[i] = group_idx;
            if sign { sign_bits[i / 8] |= 1 << (i % 8); }
            group_tails[group_idx as usize].push(tail_int);
        }

        // ── Step 2: Truncation with averaging (Zero-allocation optimization) ──
        let mut global_loss_sum = 0.0f32;
        let mut global_loss_count = 0usize;
        let mut current_tail_digits = initial_tail_digits;
        let mut round_ups = vec![Vec::with_capacity(truncate_rounds); prefixes.len()];

        for _round in 0..truncate_rounds {
            let current_divisor = 10f32.powi((prefix_digits + current_tail_digits) as i32);
            let next_divisor = 10f32.powi((prefix_digits + current_tail_digits - 1) as i32);

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

                for tail in gt.iter_mut() {
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

        // ── Step 3: Find unique tails + count ──
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

        // ── Step 4: Build tail_idx (AVX-512 with safe fallback) ──
        let mut tail_idx = vec![0u16; n];

        if is_x86_feature_detected!("avx512f") && n % 16 == 0 {
            let mut out_tail_ints = vec![0u32; n];
            unsafe {
                avx512_reconstruct_tails(
                    weights,
                    prefix_scale,
                    tail_scale,
                    &prefix_idx,
                    &round_ups,
                    truncate_rounds,
                    &mut out_tail_ints,
                );
            }
            for i in 0..n {
                let tv = out_tail_ints[i] as u16;
                tail_idx[i] = tail_idx_map.get(&tv).copied().unwrap_or(0);
            }
        } else {
            // Scalar fallback path
            for i in 0..n {
                let abs_w = weights[i].abs();
                let prefix_int = (abs_w * prefix_scale).floor() as u16;
                let prefix_val = prefix_int as f32 / prefix_scale;
                let tail_val = abs_w - prefix_val;
                let mut tail_int = (tail_val * tail_scale).round() as u32;
                let gidx = prefix_idx[i] as usize;
                for round in 0..truncate_rounds {
                    let ru = round_ups[gidx].get(round).copied().unwrap_or(false);
                    tail_int = if ru { tail_int / 10 + 1 } else { tail_int / 10 };
                }
                let tv = tail_int as u16;
                tail_idx[i] = tail_idx_map.get(&tv).copied().unwrap_or(0);
            }
        }

        let prefix_counts: Vec<u32> = (0..prefixes.len())
            .map(|gidx| group_tails[gidx].len() as u32)
            .collect();

        let tensor = Self {
            prefixes, prefix_counts, unique_tails,
            count: n, prefix_digits,
            tail_digits: current_tail_digits,
            avg_precision_lost: global_avg_lost,
        };

        let tail_width: u8 = if truncate_rounds >= 3 { 0 } else { 1 };
        let sandbag = Sandbag { prefix_idx, tail_idx, tail_width, sign_bits, count: n };
        (tensor, sandbag)
    }

    pub fn compress_from_gpu(
        weights: &[f32],
        prefix_bits: &[u32],
        tails: &[u32],
        signs: &[u32],
        prefix_digits: usize,
        truncate_rounds: usize,
    ) -> (Self, Sandbag) {
        let initial_tail_digits = Self::TOTAL_DIGITS - prefix_digits;
        let prefix_scale = 10f32.powi(prefix_digits as i32);
        let n = weights.len();

        let mut prefix_map: AHashMap<u16, u16> = AHashMap::with_capacity(512);
        let mut prefixes: Vec<u16> = Vec::new();
        let mut prefix_idx = vec![0u16; n];
        let mut sign_bits = vec![0u8; (n + 7) / 8];
        let mut group_tails: Vec<Vec<u32>> = Vec::new();

        for i in 0..n {
            let pv = f32::from_bits(prefix_bits[i]);
            let prefix_int = (pv * prefix_scale).floor() as u16;
            let group_idx = match prefix_map.get(&prefix_int) {
                Some(&idx) => idx,
                None => {
                    let idx = prefix_map.len() as u16;
                    prefix_map.insert(prefix_int, idx);
                    prefixes.push(prefix_int);
                    group_tails.push(Vec::new());
                    idx
                }
            };

            prefix_idx[i] = group_idx;
            if signs[i] != 0 { sign_bits[i / 8] |= 1 << (i % 8); }
            group_tails[group_idx as usize].push(tails[i]);
        }

        let mut global_loss_sum = 0.0f32;
        let mut global_loss_count = 0usize;
        let mut current_tail_digits = initial_tail_digits;
        let mut round_ups = vec![Vec::with_capacity(truncate_rounds); prefixes.len()];

        for _round in 0..truncate_rounds {
            let current_divisor = 10f32.powi((prefix_digits + current_tail_digits) as i32);
            let next_divisor = 10f32.powi((prefix_digits + current_tail_digits - 1) as i32);

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

                for tail in gt.iter_mut() {
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

        // ── Step 3: Find unique tails + count ──
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

        // ── Step 4: Build tail_idx (AVX-512 with safe fallback) ──
        let mut tail_idx = vec![0u16; n];
                let tail_scale = 10f32.powi(Self::TOTAL_DIGITS as i32);

        if is_x86_feature_detected!("avx512f") && n % 16 == 0 {
            let mut out_tail_ints = vec![0u32; n];
            unsafe {
                avx512_reconstruct_tails(
                    weights,
                    prefix_scale,
                    tail_scale,
                    &prefix_idx,
                    &round_ups,
                    truncate_rounds,
                    &mut out_tail_ints,
                );
            }
            for i in 0..n {
                let tv = out_tail_ints[i] as u16;
                tail_idx[i] = tail_idx_map.get(&tv).copied().unwrap_or(0);
            }
        } else {
            // Scalar fallback path
            for i in 0..n {
                let abs_w = weights[i].abs();
                let prefix_int = (abs_w * prefix_scale).floor() as u16;
                let prefix_val = prefix_int as f32 / prefix_scale;
                let tail_val = abs_w - prefix_val;
                let mut tail_int = (tail_val * tail_scale).round() as u32;
                let gidx = prefix_idx[i] as usize;
                for round in 0..truncate_rounds {
                    let ru = round_ups[gidx].get(round).copied().unwrap_or(false);
                    tail_int = if ru { tail_int / 10 + 1 } else { tail_int / 10 };
                }
                let tv = tail_int as u16;
                tail_idx[i] = tail_idx_map.get(&tv).copied().unwrap_or(0);
            }
        }

        let prefix_counts: Vec<u32> = (0..prefixes.len())
            .map(|gidx| group_tails[gidx].len() as u32)
            .collect();

        let tensor = Self {
            prefixes, prefix_counts, unique_tails,
            count: n, prefix_digits,
            tail_digits: current_tail_digits,
            avg_precision_lost: global_avg_lost,
        };

        let tail_width: u8 = if truncate_rounds >= 3 { 0 } else { 1 };
        let sandbag = Sandbag { prefix_idx, tail_idx, tail_width, sign_bits, count: n };
        (tensor, sandbag)
    }    
    
    pub fn decompress_all(&self, sandbag: &Sandbag) -> Vec<f32> {
        let prefix_scale = 10f32.powi(self.prefix_digits as i32);
        let divisor = 10f32.powi((self.prefix_digits + self.tail_digits) as i32);
        let mut result = Vec::with_capacity(self.count);

        for i in 0..self.count {
            let p_idx = sandbag.prefix_idx.get(i).copied().unwrap_or(0) as usize;
            let t_idx = sandbag.tail_idx.get(i).copied().unwrap_or(0) as usize;

            let prefix_int = self.prefixes.get(p_idx).copied().unwrap_or(0);
            let prefix = prefix_int as f32 / prefix_scale;
            let tail = self.unique_tails.get(t_idx).map(|ut| ut.value).unwrap_or(0);

            let mut value = prefix + tail as f32 / divisor;
            value += self.avg_precision_lost;

            let sign = (sandbag.sign_bits.get(i / 8).copied().unwrap_or(0) >> (i % 8)) & 1 != 0;
            result.push(if sign { -value } else { value });
        }
        result
    }

    pub fn decompress_all_global(&self, sandbag: &Sandbag, global: &GlobalTable) -> Vec<f32> {
        let tail_divisor = 10f32.powi((global.prefix_digits + self.tail_digits) as i32);
        let gt_scale = 10f32.powi(global.prefix_digits as i32);
        let scale_diff = global.prefix_digits as i32 - self.prefix_digits as i32;

        let mut prefix_lookup: AHashMap<u16, usize> = AHashMap::with_capacity(self.prefixes.len());
        for &pv in &self.prefixes {
            let norm = if scale_diff > 0 {
                (pv as u32) * 10u32.pow(scale_diff as u32)
            } else {
                pv as u32
            };
            if !prefix_lookup.contains_key(&pv) {
                if let Some(gi) = global.prefixes.iter().position(|&gp| (gp * gt_scale).round() as u32 == norm) {
                    prefix_lookup.insert(pv, gi);
                }
            }
        }

        let mut tail_lookup: AHashMap<(u16, u16), usize> = AHashMap::new();
        for (cp_idx, &pv) in self.prefixes.iter().enumerate() {
            let gp_idx = *prefix_lookup.get(&pv).unwrap_or(&0);
            let global_tails = &global.tails_for_prefix[gp_idx];
            let mut val_to_gi: AHashMap<u32, usize> = AHashMap::with_capacity(global_tails.len());
            for (gi, &t) in global_tails.iter().enumerate() {
                val_to_gi.insert(t, gi);
            }
            for (ct_idx, ut) in self.unique_tails.iter().enumerate() {
                if let Some(&gi) = val_to_gi.get(&(ut.value as u32)) {
                    tail_lookup.insert((cp_idx as u16, ct_idx as u16), gi);
                }
            }
        }

        let mut result = Vec::with_capacity(self.count);
        for i in 0..self.count {
            let p_idx = sandbag.prefix_idx.get(i).copied().unwrap_or(0);
            let t_idx = sandbag.tail_idx.get(i).copied().unwrap_or(0) as u16;

            let pv = self.prefixes.get(p_idx as usize).copied().unwrap_or(0);
            let gp = *prefix_lookup.get(&pv).unwrap_or(&0);
            let gt = *tail_lookup.get(&(p_idx, t_idx)).unwrap_or(&0);

            let prefix = global.prefixes[gp];
            let tail = global.tails_for_prefix[gp][gt];

            let mut value = prefix + tail as f32 / tail_divisor;
            value += self.avg_precision_lost;

            let sign = (sandbag.sign_bits.get(i / 8).copied().unwrap_or(0) >> (i % 8)) & 1 != 0;
            result.push(if sign { -value } else { value });
        }
        result
    }

    pub fn compressed_bytes(&self) -> usize {
        let header = 4 + 4 + 4 + 4 + 4; 
        let front = self.prefixes.len() * 2 + self.unique_tails.len() * 2;     
        let flags = 3; 
        header + front + flags
    }
}

// ============================================================================
// ── SECTION 3: AVX-512 CORE OPTIMIZATION VECTOR PROCESSING KERNEL ───────────
// ============================================================================

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

        let v_w = _mm512_abs_ps(_mm512_loadu_ps(weights.as_ptr().add(offset)));
        
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
            let mask_even_odd = _mm512_set_epi32(
                0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1
            );
            let v_div = _mm512_mask_blend_epi32(
                0xAAAA, // Mask selecting odd lanes only
                div_even,
                _mm512_slli_epi64(div_odd, 32)
            );

            let v_div_plus_one = _mm512_add_epi32(v_div, v_one);
            
            // Masked blend: lanes with mask=1 get (tail/10 + 1), mask=0 get (tail/10)
            v_tail_int = _mm512_mask_blend_epi32(mask, v_div, v_div_plus_one);
        }

        _mm512_storeu_si512(out_tail_ints.as_mut_ptr().add(offset) as *mut _, v_tail_int);
    }
}

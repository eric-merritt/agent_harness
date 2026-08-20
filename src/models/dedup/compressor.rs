use crate::models::dedup::tensor::DedupCountTensor;
use hashbrown::HashMap as AHashMap;
use crate::models::dedup::types::{UniqueTail, Sandbag};
use crate::models::dedup::avx512_kernel;

impl DedupCountTensor {

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
                avx512_kernel::avx512_reconstruct_tails(
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
                avx512_kernel::avx512_reconstruct_tails(
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
}
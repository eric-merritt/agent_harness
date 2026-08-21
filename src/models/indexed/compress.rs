use super::tensor::CountIndexedTensor;
use std::collections::HashMap;

impl CountIndexedTensor {

/// Compress weights into count-indexed layout with iterative TAIL truncation.
    ///
    /// The prefix stays at FULL precision — never truncated.
    /// Only the tail is iteratively shortened:
    ///   Round 1: tail=2899, last=9 (≥5), round up 2nd-to-last → 290, drop → 290
    ///   Round 2: tail=290, last=0 (<5), drop → 29
    ///   Round 3: tail=29, last=9 (≥5), round up → 3, drop → 3
    /// After each round, shorter tails are more likely to chain to other prefixes.
    pub fn compress(weights: &[f32], prefix_digits: usize, truncate_rounds: usize) -> Self {
        let scale = 10f32.powi(prefix_digits as i32);

        let mut prefix_map: HashMap<u32, usize> = HashMap::new();
        let mut prefixes: Vec<f32> = Vec::new();
        let mut groups: Vec<Vec<(u8, bool, bool)>> = Vec::new();

        for &w in weights {
            let sign = w < 0.0;
            let abs_w = w.abs();

            // Prefix at FULL precision — never touched
            let prefix = (abs_w * scale).floor() / scale;
            let mut tail_val = abs_w - prefix;

            // Iteratively truncate ONLY the tail
            let mut rounded_up = false;
            let mut tail_digits = 7 - prefix_digits; // f32 has ~7 significant digits
            for _ in 0..truncate_rounds {
                if tail_digits == 0 { break; }
                let tail_precision = 10f32.powi(tail_digits as i32);
                let digits = (tail_val * tail_precision).round() as i64;
                let last = (digits % 10) as u8;
                let remaining = digits / 10;
                tail_digits -= 1;
                let new_precision = 10f32.powi(tail_digits as i32);
                tail_val = if last >= 5 {
                    rounded_up = true;
                    (remaining + 1) as f32 / new_precision
                } else {
                    remaining as f32 / new_precision
                };
            }

            // Normalize truncated tail to u8
            let tail_byte = (tail_val * scale * 255.0).round().clamp(0.0, 255.0) as u8;

            let prefix_bits = prefix.to_bits();
            let group_idx = if let Some(&idx) = prefix_map.get(&prefix_bits) {
                idx
            } else if prefix_map.len() < 256 {
                let idx = prefix_map.len();
                prefix_map.insert(prefix_bits, idx);
                prefixes.push(prefix);
                groups.push(Vec::new());
                idx
            } else {
                0
            };

            groups[group_idx].push((tail_byte, sign, rounded_up));
        }

        // Phase 2: Flatten into count-indexed layout
        let mut tails: Vec<u8> = Vec::new();
        let mut counts: Vec<u32> = Vec::with_capacity(prefixes.len());
        let mut sign_bits: Vec<u8> = Vec::new();
        let mut round_bits: Vec<u8> = Vec::new();
        let mut sign_byte: u8 = 0;
        let mut round_byte: u8 = 0;
        let mut bit_pos: u8 = 0;

        for group in &groups {
            counts.push(group.len() as u32);
            for &(tail, sign, rounded_up) in group {
                tails.push(tail);

                // Pack sign bit
                if sign { sign_byte |= 1 << bit_pos; }
                // Pack round bit
                if rounded_up { round_byte |= 1 << bit_pos; }
                bit_pos += 1;
                if bit_pos == 8 {
                    sign_bits.push(sign_byte);
                    round_bits.push(round_byte);
                    sign_byte = 0;
                    round_byte = 0;
                    bit_pos = 0;
                }
            }
        }
        // Flush remaining bits
        if bit_pos > 0 {
            sign_bits.push(sign_byte);
            round_bits.push(round_byte);
        }

        Self {
            prefixes,
            tails,
            counts,
            sign_bits,
            round_bits,
            tail_scale: scale,
            count: weights.len(),
        }
    }

    pub fn comp_ratio(&self) -> f32 {
        let comp = self.compressed_bytes() as f32;
        if comp == 0.0 { 1.0 } else { self.original_bytes() as f32 / comp }
    }

    pub fn original_bytes(&self) -> usize {
        &self.count * 4
    }

    
}
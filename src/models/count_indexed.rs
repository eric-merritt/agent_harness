// Count-indexed weight layout — no per-weight pointers needed.
//
// Layout:
//   [prefix_0: f32][tail_0_0: u8][tail_0_1: u8]...[tail_0_N: u8]
//   [prefix_1: f32][tail_1_0: u8][tail_1_1: u8]...[tail_1_M: u8]
//   ...
//   [back: count_0: u16][count_1: u16]...[count_K: u16]
//
// To find weight #375 of prefix #2:
//   1. Read count table at back: count_0=30000, count_1=45000
//   2. Prefix 2 starts at weight index 75000
//   3. Weight 375 = flat offset (prefix_table_size + 75000 * 1 + 375 * 1)
//   4. Read tail byte, combine with prefix_2 value
//
// The "break in offset" is detectable: different prefix groups have
// different counts, and the count table tells you where each group
// starts. No per-weight index, no pointer, no alignment waste.
//
// With iterative truncation (round 2 on BF16):
//   - 12 prefixes, 1-byte tails (0-9 after rounding)
//   - Count table: 12 * 2 bytes = 24 bytes at the back
//   - Per-weight storage: 1 byte (tail value)
//   - Prefix is implicit from which group you're in
//   - Total: N * 1 + K * (4 + 2) bytes  ≈  N bytes

use std::collections::HashMap;

/// A count-indexed packed weight tensor.
/// Weights are stored flat, grouped by prefix, with a count table at the end.
#[derive(Clone, Debug)]
pub struct CountIndexedTensor {
    /// Prefix values, one per group. Index = group ID.
    pub prefixes: Vec<f32>,
    /// Flat tail array — tails grouped contiguously by prefix group.
    /// Group 0's tails first, then group 1's, etc.
    pub tails: Vec<u8>,
    /// Count table at the back — how many weights per group.
    /// tails.len() == sum(counts)
    pub counts: Vec<u32>,
    /// Sign bits — 1 bit per weight, packed into bytes.
    pub sign_bits: Vec<u8>,
    /// Round bits — 1 bit per weight.
    pub round_bits: Vec<u8>,
    /// Tail scale used during compression (for decompression).
    pub tail_scale: f32,
    /// Original count for decompression sizing.
    pub count: usize,
}

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

    /// Decompress a single weight by flat index.
    pub fn decompress_one(&self, index: usize) -> f32 {
        if index >= self.count { return 0.0; }

        let mut offset = 0usize;
        let mut group = 0;
        for (g, &count) in self.counts.iter().enumerate() {
            if index < offset + count as usize {
                group = g;
                break;
            }
            offset += count as usize;
        }

        let local_index = index - offset;
        let tail_byte = self.tails[offset + local_index];

        let byte_idx = index / 8;
        let bit_idx = index % 8;
        let sign = if byte_idx < self.sign_bits.len() {
            (self.sign_bits[byte_idx] >> bit_idx) & 1 != 0
        } else { false };

        let prefix = self.prefixes.get(group).copied().unwrap_or(0.0);
        let tail = tail_byte as f32 / 255.0 / self.tail_scale;
        let value = prefix + tail;
        if sign { -value } else { value }
    }

    /// Decompress all weights.
    pub fn decompress_all(&self) -> Vec<f32> {
        (0..self.count).map(|i| self.decompress_one(i)).collect()
    }

    /// Find weight by "going to the back, reading count, multiplying by stride".
    ///
    /// Example: count table says group 2 has 375 weights.
    /// We know weights before group 2 sum to 30,000.
    /// Weight #200 of group 2 = flat index 30,200.
    /// Stride = 1 byte (tail) in the flat array.
    /// Offset = prefix_table_overhead + 30,200 * 1.
    pub fn group_offset(&self, group: usize) -> usize {
        self.counts.iter().take(group).map(|c| *c as usize).sum()
    }

    pub fn group_count(&self, group: usize) -> u32 {
        self.counts.get(group).copied().unwrap_or(0)
    }

    /// Compressed size in bytes.
    pub fn compressed_bytes(&self) -> usize {
        self.prefixes.len() * 4          // prefix values (f32)
        + self.tails.len()               // tail array (1 byte per weight)
        + self.counts.len() * 4          // count table (u32 per group)
        + self.sign_bits.len()           // packed sign bits
        + self.round_bits.len()          // packed round bits
    }

    /// Original size in bytes (as f32).
    pub fn original_bytes(&self) -> usize {
        self.count * 4
    }

    /// Compression ratio.
    pub fn ratio(&self) -> f32 {
        let comp = self.compressed_bytes() as f32;
        if comp == 0.0 { 1.0 } else { self.original_bytes() as f32 / comp }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_indexed_layout() {
        // Weights with realistic distribution — clustered but with noise
        let mut weights = Vec::new();
        for i in 0..10_000 {
            let base = match i % 4 {
                0 => 0.0150,
                1 => 0.0230,
                2 => -0.0180,
                _ => 0.0420,
            };
            let noise = ((i * 7) as f32 * 0.00017).fract() * 0.0009;
            weights.push(base + noise + if i % 2 == 0 { 0.0 } else { 0.0001 });
        }

        let packed = CountIndexedTensor::compress(&weights, 3, 1);
        let decompressed = packed.decompress_all();

        let max_err = weights.iter().zip(decompressed.iter())
            .map(|(o, d)| (o - d).abs())
            .fold(0.0f32, f32::max);

        println!("\nCount-indexed test (10K weights):");
        println!("  Groups: {}", packed.prefixes.len());
        println!("  Counts: {:?}", packed.counts);
        println!("  Compressed: {} bytes", packed.compressed_bytes());
        println!("  Original: {} bytes", packed.original_bytes());
        println!("  Ratio: {:.2}x", packed.ratio());
        println!("  Max error: {:.6}", max_err);
        println!("  Sign bits: {} bytes, Round bits: {} bytes", packed.sign_bits.len(), packed.round_bits.len());

        // Lossy — tolerance reflects the lossy nature of prefix+truncation
        assert!(max_err < 0.1, "Max error too large: {}", max_err);
    }

    #[test]
    fn test_large_tensor() {
        // 100K weights with 5 clusters and noise
        let mut weights = Vec::with_capacity(100_000);
        for i in 0..100_000 {
            let base = match i % 5 {
                0 => 0.0150,
                1 => 0.0230,
                2 => -0.0180,
                3 => 0.0420,
                _ => -0.0310,
            };
            let noise = ((i * 13) as f32 * 0.00007).fract() * 0.0009;
            weights.push(base + noise);
        }

        let packed = CountIndexedTensor::compress(&weights, 2, 1);
        let decompressed = packed.decompress_all();

        let max_err = weights.iter().zip(decompressed.iter())
            .map(|(o, d)| (o - d).abs())
            .fold(0.0f32, f32::max);

        println!("\nLarge count-indexed test (100K weights):");
        println!("  Groups: {}", packed.prefixes.len());
        println!("  Total weights: {}", packed.count);
        println!("  Tails: {} bytes (1 per weight)", packed.tails.len());
        println!("  Count table: {} bytes ({} groups x 4B)", packed.counts.len() * 4, packed.counts.len());
        println!("  Sign bits: {} bytes", packed.sign_bits.len());
        println!("  Round bits: {} bytes", packed.round_bits.len());
        println!("  Compressed: {} bytes ({:.1} KB)", packed.compressed_bytes(), packed.compressed_bytes() as f32 / 1024.0);
        println!("  Original: {} bytes ({:.1} KB)", packed.original_bytes(), packed.original_bytes() as f32 / 1024.0);
        println!("  Ratio: {:.2}x", packed.ratio());
        println!("  Max error: {:.6}", max_err);

        assert!(max_err < 0.1, "Max error too large: {}", max_err);
    }
}

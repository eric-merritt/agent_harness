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
    
    /// Find weight by "going to the back, reading count, multiplying by stride"
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
        println!("  Ratio: {:.2}x", packed.comp_ratio());
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
        println!("  Ratio: {:.2}x", packed.comp_ratio());
        println!("  Max error: {:.6}", max_err);

        assert!(max_err < 0.1, "Max error too large: {}", max_err);
    }
}

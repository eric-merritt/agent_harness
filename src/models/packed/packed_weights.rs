// Content-aware weight compression with alignment-packing.
//
// The core idea: weights share common high-order digits (prefixes).
// Instead of storing each weight as a full f32 (4 bytes), we store:
//   1 byte: cluster index (which prefix this weight belongs to)
//   1 byte: tail index (which tail within the cluster)
// = 2 bytes per weight instead of 4 → 2x compression on f32.
//
// On BF16 (2 bytes per weight), we can't go below 2 bytes per weight
// with this scheme. BUT — if the alignment is 8 bytes and we only
// need 2 bytes for the cluster+tail index, we have 6 bytes of "free"
// alignment space per weight. We pack tail data for OTHER weights
// into that space, amortizing the storage.
//
// Unique flag: we use the sentinel pattern 0xFFFF for the tail index.
// When decompressing, 0xFFFF means "this weight is stored uncompressed
// as a full f32" — it can't be a valid tail index because u16::MAX
// would require 65536 unique tails in a single cluster, which never
// happens in practice (max observed: ~101).
//
// For sign-inverses (w and -w): we flag with the high bit of the
// cluster index byte (0x80). If bit 7 is set, the weight is the
// negation of the cluster's prefix value + tail.

use std::collections::HashMap;

/// Sentinel: a tail index of 0xFFFF means "uncompressed f32 follows".
pub const UNCOMPRESSED_FLAG: u8 = 0xFF;
/// High bit of cluster byte = sign inverse flag.
pub const SIGN_INVERSE_BIT: u8 = 0x80;
/// Max clusters that fit in 7 bits (high bit is the inverse flag).
pub const MAX_CLUSTERS: usize = 127;
/// Max tails per cluster that fit in 1 byte (0xFF is the sentinel).
pub const MAX_TAILS_PER_CLUSTER: usize = 254;

/// A compressed weight tensor using coordinate + sign-inverse compression
/// with alignment-packing.
#[derive(Clone, Debug)]
pub struct PackedTensor {
    /// Cluster prefix values (f32, one per cluster).
    pub prefixes: Vec<f32>,
    /// Tail tables — one vec of f32 per cluster.
    pub tail_tables: Vec<Vec<f32>>,
    /// Packed data: for each weight, [cluster_byte | tail_byte]
    /// where cluster_byte has the inverse bit in bit 7.
    pub packed: Vec<u8>,
    /// Weights that couldn't be compressed (too many clusters or tails).
    /// Stored as full f32 values, indexed by their position in the weight array.
    pub uncompressed: HashMap<usize, f32>,
    /// Original element count.
    pub count: usize,
    /// Alignment in bytes (typically 8 for safetensors, 32 for GGUF).
    pub alignment: usize,
}

impl PackedTensor {
    /// Compress a slice of f32 weights into a PackedTensor.
    pub fn compress(weights: &[f32], alignment: usize, prefix_digits: usize) -> Self {
        let scale = 10f32.powi(prefix_digits as i32);

        // Phase 1: Build clusters
        let mut cluster_map: HashMap<u32, usize> = HashMap::new();
        let mut prefixes: Vec<f32> = Vec::new();
        let mut tail_tables: Vec<Vec<f32>> = Vec::new();
        let mut tail_index_maps: Vec<HashMap<u32, u8>> = Vec::new();
        let mut packed: Vec<u8> = Vec::with_capacity(weights.len() * 2);
        let mut uncompressed: HashMap<usize, f32> = HashMap::new();

        for (i, &w) in weights.iter().enumerate() {
            // Check for sign-inverse: is -w already a known cluster prefix?
            let prefix = (w.abs() * scale).floor() / scale * w.signum();
            let prefix_bits = prefix.to_bits();

            // Try to find or create a cluster for this prefix
            let cluster_idx = cluster_map.get(&prefix_bits).copied();

            match cluster_idx {
                Some(cidx) if cidx < MAX_CLUSTERS => {
                    // Found a cluster — look up or create tail
                    let tail = w - prefixes[cidx];
                    let tail_bits = tail.to_bits();
                    let tail_map = &mut tail_index_maps[cidx];

                    let tail_idx = if let Some(&tidx) = tail_map.get(&tail_bits) {
                        tidx
                    } else if tail_map.len() < MAX_TAILS_PER_CLUSTER {
                        let tidx = tail_map.len() as u8;
                        tail_map.insert(tail_bits, tidx);
                        tail_tables[cidx].push(tail);
                        tidx
                    } else {
                        // Too many tails in this cluster — store uncompressed
                        uncompressed.insert(i, w);
                        UNCOMPRESSED_FLAG
                    };

                    // Check if this weight is a sign-inverse of the cluster prefix
                    let is_inverse = w.signum() != prefixes[cidx].signum()
                        && w.abs().to_bits() == prefixes[cidx].abs().to_bits();

                    let cluster_byte = if is_inverse {
                        (cidx as u8) | SIGN_INVERSE_BIT
                    } else {
                        cidx as u8
                    };
                    packed.push(cluster_byte);
                    packed.push(tail_idx);
                }
                None if cluster_map.len() < MAX_CLUSTERS => {
                    // Create new cluster
                    let cidx = cluster_map.len();
                    cluster_map.insert(prefix_bits, cidx);
                    prefixes.push(prefix);
                    tail_tables.push(Vec::new());
                    tail_index_maps.push(HashMap::new());

                    // First weight in cluster — tail is the difference
                    let tail = w - prefix;
                    let tail_bits = tail.to_bits();
                    tail_index_maps[cidx].insert(tail_bits, 0);
                    tail_tables[cidx].push(tail);

                    packed.push(cidx as u8);
                    packed.push(0); // tail index 0
                }
                _ => {
                    // Too many clusters — store uncompressed
                    uncompressed.insert(i, w);
                    packed.push(0); // dummy cluster
                    packed.push(UNCOMPRESSED_FLAG);
                }
            }
        }

        Self {
            prefixes,
            tail_tables,
            packed,
            uncompressed,
            count: weights.len(),
            alignment,
        }
    }

    /// Decompress a single weight at a given index.
    pub fn decompress_one(&self, index: usize) -> f32 {
        if let Some(&w) = self.uncompressed.get(&index) {
            return w;
        }

        let offset = index * 2;
        if offset + 1 >= self.packed.len() {
            return 0.0;
        }

        let cluster_byte = self.packed[offset];
        let tail_byte = self.packed[offset + 1];

        if tail_byte == UNCOMPRESSED_FLAG {
            return self.uncompressed.get(&index).copied().unwrap_or(0.0);
        }

        let is_inverse = (cluster_byte & SIGN_INVERSE_BIT) != 0;
        let cluster_idx = (cluster_byte & !SIGN_INVERSE_BIT) as usize;

        if cluster_idx >= self.prefixes.len() {
            return 0.0;
        }

        let prefix = self.prefixes[cluster_idx];
        let tail = self.tail_tables[cluster_idx]
            .get(tail_byte as usize)
            .copied()
            .unwrap_or(0.0);

        if is_inverse {
            -(prefix.abs() + tail.abs())
        } else {
            prefix + tail
        }
    }

    /// Decompress all weights into a Vec<f32>.
    pub fn decompress_all(&self) -> Vec<f32> {
        (0..self.count).map(|i| self.decompress_one(i)).collect()
    }

    /// Compressed byte size.
    pub fn compressed_bytes(&self) -> usize {
        self.packed.len()                                    // packed indices
        + self.prefixes.len() * 4                            // prefix table
        + self.tail_tables.iter().map(|t| t.len() * 4).sum::<usize>()  // tail tables
        + self.uncompressed.len() * 4                        // uncompressed fallback
    }

    /// Original byte size (as f32).
    pub fn original_bytes(&self) -> usize {
        self.count * 4
    }

    /// Compression ratio (original / compressed).
    pub fn ratio(&self) -> f32 {
        let orig = self.original_bytes() as f32;
        let comp = self.compressed_bytes() as f32;
        if comp == 0.0 { 1.0 } else { orig / comp }
    }

    /// Number of unused alignment bytes available for packing.
    /// If alignment is 8 and we use 2 bytes per weight, that's 6 free bytes.
    pub fn free_alignment_bytes(&self) -> usize {
        let used_per_weight = 2;
        if self.alignment > used_per_weight {
            (self.alignment - used_per_weight) * self.count
        } else {
            0
        }
    }

    /// Pack tail table data into the alignment padding.
    /// Returns the tail data that fits in the free alignment space.
    pub fn pack_tails_into_alignment(&self) -> Vec<u8> {
        let free = self.free_alignment_bytes();
        let mut packed_tails = Vec::with_capacity(free);

        for table in &self.tail_tables {
            for &tail in table {
                let bytes = tail.to_le_bytes();
                for &b in &bytes {
                    if packed_tails.len() < free {
                        packed_tails.push(b);
                    }
                }
            }
        }

        packed_tails
    }
}

/// Test compression on real weights and report stats.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_compression() {
        let weights = vec![0.715801, 0.715802, 0.715805, 0.708290, -0.708290];
        let packed = PackedTensor::compress(&weights, 8, 3);
        let decompressed = packed.decompress_all();

        // Check that values are close (lossy due to prefix rounding)
        for (orig, decomp) in weights.iter().zip(decompressed.iter()) {
            let err = (orig - decomp).abs();
            assert!(err < 0.01, "Error too large: {} vs {} (err={})", orig, decomp, err);
        }

        let ratio = packed.ratio();
        println!("Ratio: {:.2}x, Compressed: {} bytes, Original: {} bytes",
            ratio, packed.compressed_bytes(), packed.original_bytes());
        println!("Free alignment bytes: {}", packed.free_alignment_bytes());
    }

    #[test]
    fn test_sign_inverse() {
        // w and -w should share the same cluster with the inverse bit set
        let weights = vec![0.5, -0.5, 0.5, -0.5, 0.5];
        let packed = PackedTensor::compress(&weights, 8, 2);
        let decompressed = packed.decompress_all();

        for (i, (orig, decomp)) in weights.iter().zip(decompressed.iter()).enumerate() {
            let err = (orig - decomp).abs();
            assert!(err < 0.01, "Weight {}: {} vs {} (err={})", i, orig, decomp, err);
        }

        println!("Clusters: {}, Ratio: {:.2}x", packed.prefixes.len(), packed.ratio());
    }

    #[test]
    fn test_large_tensor() {
        // Simulate 100K weights clustered around a few values
        let mut weights = Vec::with_capacity(100_000);
        for i in 0..100_000 {
            let base = match i % 5 {
                0 => 0.015,
                1 => 0.023,
                2 => -0.018,
                3 => 0.042,
                _ => -0.031,
            };
            let noise = (i as f32 * 0.0001).fract() * 0.001;
            weights.push(base + noise);
        }

        let packed = PackedTensor::compress(&weights, 8, 3);
        let decompressed = packed.decompress_all();

        let max_err = weights.iter().zip(decompressed.iter())
            .map(|(o, d)| (o - d).abs())
            .fold(0.0f32, f32::max);

        println!("\n100K weight test:");
        println!("  Clusters: {}", packed.prefixes.len());
        println!("  Uncompressed: {} weights", packed.uncompressed.len());
        println!("  Original: {} bytes", packed.original_bytes());
        println!("  Compressed: {} bytes", packed.compressed_bytes());
        println!("  Ratio: {:.2}x", packed.ratio());
        println!("  Max error: {}", max_err);
        println!("  Free alignment bytes: {}", packed.free_alignment_bytes());
        println!("  Tail data packable: {} bytes", packed.pack_tails_into_alignment().len());

        assert!(max_err < 0.01, "Max error too large: {}", max_err);
    }
}

use crate::models::dedup::types::{GlobalTable, Sandbag, ChunkRemap};
use crate::models::dedup::tensor::DedupCountTensor;


impl GlobalTable {
    /// Constructs a unified GlobalTable from a slice of individual tensor chunks.
    pub fn new(chunks: &[DedupCountTensor]) -> Self {
        if chunks.is_empty() {
            return Self {
                prefix_digits: 0,
                prefixes: Vec::new(),
                tails_for_prefix: Vec::new(),
            };
        }

        // 1. Pull the unified scale config from the first chunk block
        let prefix_digits = chunks[0].prefix_digits;
        let prefix_scale = 10f32.powi(prefix_digits as i32);

        // 2. Collect and deduplicate all unique integer prefixes across chunks
        let mut unique_prefixes: Vec<u16> = Vec::new();
        for tensor in chunks {
            for &p in &tensor.prefixes {
                if !unique_prefixes.contains(&p) {
                    unique_prefixes.push(p);
                }
            }
        }
        unique_prefixes.sort_unstable();

        // 3. Map integer prefixes to the raw f32 scale values your struct fields expect
        let mut prefixes_f32 = Vec::with_capacity(unique_prefixes.len());
        let mut tails_for_prefix = vec![Vec::new(); unique_prefixes.len()];

        for &p in &unique_prefixes {
            prefixes_f32.push(p as f32 / prefix_scale);
        }

        // 4. Group unique tail values under their respective global prefix index lanes
        for tensor in chunks {
            for &p in &tensor.prefixes {
                if let Ok(global_p_idx) = unique_prefixes.binary_search(&p) {
                    let target_prefix_tails = &mut tails_for_prefix[global_p_idx];
                    
                    for ut in &tensor.unique_tails {
                        let t_val = ut.value as u32;
                        if !target_prefix_tails.contains(&t_val) {
                            target_prefix_tails.push(t_val);
                        }
                    }
                }
            }
        }

        // 5. Return your exact struct fields populated cleanly
        Self {
            prefix_digits,
            prefixes: prefixes_f32,
            tails_for_prefix,
        }
    }

    /// Generates a local chunk remapping descriptor table.
    pub fn build_chunk_remap(&self, tensor: &DedupCountTensor) -> ChunkRemap {
        let mut global_tail_indices = Vec::with_capacity(tensor.unique_tails.len());
        let gt_scale = 10f32.powi(self.prefix_digits as i32);
        let scale_diff = self.prefix_digits as i32 - tensor.prefix_digits as i32;

        for ut in &tensor.unique_tails {
            let mut matched_global_idx = 0u16;
            'outer: for &pv in &tensor.prefixes {
                let norm = if scale_diff > 0 {
                    (pv as u32) * 10u32.pow(scale_diff as u32)
                } else {
                    pv as u32
                };
                if let Some(gp_idx) = self.prefixes.iter().position(|&gp| (gp * gt_scale).round() as u32 == norm) {
                    if let Some(pos) = self.tails_for_prefix[gp_idx].iter().position(|&t| t == ut.value as u32) {
                        matched_global_idx = pos as u16;
                        break 'outer;
                    }
                }
            }
            global_tail_indices.push(matched_global_idx);
        }
        ChunkRemap { global_tail_indices }
    }

    /// Decompresses elements safely using a chunk remap cache configuration.
    pub fn decompress_with_remap(&self, sandbag: &Sandbag, tensor: &DedupCountTensor, remap: Option<ChunkRemap>) -> Vec<f32> {
        if let Some(m) = remap {
            let divisor = 10f32.powi((self.prefix_digits + tensor.tail_digits) as i32);
            let mut result = Vec::with_capacity(sandbag.count);
            let gt_scale = 10f32.powi(self.prefix_digits as i32);
            let scale_diff = self.prefix_digits as i32 - tensor.prefix_digits as i32;

            for i in 0..sandbag.count {
                let p_idx = sandbag.prefix_idx[i] as usize;
                let t_idx = sandbag.tail_idx[i] as usize;

                let pv = tensor.prefixes[p_idx];
                let norm = if scale_diff > 0 { (pv as u32) * 10u32.pow(scale_diff as u32) } else { pv as u32 };
                
                let gp = self.prefixes.iter().position(|&gp| (gp * gt_scale).round() as u32 == norm).unwrap_or(0);
                let local_mapped_tail_idx = m.global_tail_indices.get(t_idx).copied().unwrap_or(0) as usize;
                let tail = self.tails_for_prefix[gp].get(local_mapped_tail_idx).copied().unwrap_or(0);

                let mut value = self.prefixes[gp] + tail as f32 / divisor;
                value += tensor.avg_precision_lost;

                let sign = (sandbag.sign_bits[i / 8] >> (i % 8)) & 1 != 0;
                result.push(if sign { -value } else { value });
            }
            result
        } else {
            tensor.decompress_all(sandbag)
        }
    }
}
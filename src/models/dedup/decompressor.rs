use crate::models::dedup::tensor::DedupCountTensor;
use hashbrown::HashMap as AHashMap;
use crate::models::dedup::types::{Sandbag, GlobalTable};

impl DedupCountTensor {

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
}
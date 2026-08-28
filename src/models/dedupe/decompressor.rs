use crate::models::dedupe::tensor::DedupCountTensor;
use crate::models::dedupe::types::{GlobalTable, Sandbag};

impl DedupCountTensor {
	pub fn decompress_all(&self, sandbag: &Sandbag) -> Vec<f32> {
		let n = sandbag.count;
		let prefix_scale = 10f32.powi(sandbag.prefix_digits as i32);
		let tail_scale = 10_000_000.0f32;
		let mut result = vec![0.0f32; n];

		// Reconstruct from manifest: (prefix_idx, tail_idx) + sign bitvector
		for (i, &(p_idx, t_idx)) in sandbag.manifest.iter().enumerate() {
			let p_int = sandbag.unique_prefixes[p_idx as usize];
			let t_int = sandbag.unique_tails[t_idx as usize];
			let abs_w = (p_int as f32) / prefix_scale + (t_int as f32) / tail_scale;
			let sign = (sandbag.signs[i / 8] >> (i % 8)) & 1;
			result[i] = if sign != 0 { -abs_w } else { abs_w };
		}

		// Restore outliers at full precision
		for &(pos, val) in &sandbag.outliers {
			if pos < n {
				result[pos] = val;
			}
		}

		result
	}

	pub fn decompress_all_global(&self, sandbag: &Sandbag, _global: &GlobalTable) -> Vec<f32> {
		// For now, use the same decompression as decompress_all
		// Global table integration can be added later for cross-tensor dedup
		self.decompress_all(sandbag)
	}
}

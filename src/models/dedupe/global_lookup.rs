use crate::models::dedupe::tensor::DedupCountTensor;
use crate::models::dedupe::types::{ChunkRemap, GlobalTable, Sandbag};

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

		let prefix_digits = chunks[0].prefix_digits;
		let prefix_scale = 10f32.powi(prefix_digits as i32);

		let mut unique_prefixes: Vec<u8> = Vec::new();
		for tensor in chunks {
			for &p in &tensor.prefixes {
				if !unique_prefixes.contains(&p) {
					unique_prefixes.push(p);
				}
			}
		}
		unique_prefixes.sort_unstable();

		let mut prefixes_f32 = Vec::with_capacity(unique_prefixes.len());
		let mut tails_for_prefix = vec![Vec::new(); unique_prefixes.len()];

		for &p in &unique_prefixes {
			prefixes_f32.push(p as f32 / prefix_scale);
		}

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
				if let Some(gp_idx) = self
					.prefixes
					.iter()
					.position(|&gp| (gp * gt_scale).round() as u32 == norm)
				{
					if let Some(pos) = self.tails_for_prefix[gp_idx]
						.iter()
						.position(|&t| t == ut.value as u32)
					{
						matched_global_idx = pos as u16;
						break 'outer;
					}
				}
			}
			global_tail_indices.push(matched_global_idx);
		}
		ChunkRemap {
			global_tail_indices,
		}
	}

	/// Decompress using the new quantized sandbag format.
	pub fn decompress_with_remap(
		&self,
		sandbag: &Sandbag,
		tensor: &DedupCountTensor,
		_remap: Option<ChunkRemap>,
	) -> Vec<f32> {
		// Just use the simple decompression for now since we changed the format
		tensor.decompress_all(sandbag)
	}
}

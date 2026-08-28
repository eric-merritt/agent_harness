use std::collections::HashMap;

use super::constants::*;
use super::flags::DataFlags::*;
use super::tensor::PackedTensor;

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
						UncompressedFlag as u8
					};

					// Check if this weight is a sign-inverse of the cluster prefix
					let is_inverse = w.signum() != prefixes[cidx].signum()
						&& w.abs().to_bits() == prefixes[cidx].abs().to_bits();

					let cluster_byte = if is_inverse {
						(cidx as u8) | SignInverseBit as u8
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
					packed.push(UncompressedFlag as u8);
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
}

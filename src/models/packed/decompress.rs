use super::flags::DataFlags::*;
use super::tensor::PackedTensor;

impl PackedTensor {
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

		if tail_byte == UncompressedFlag as u8 {
			return self.uncompressed.get(&index).copied().unwrap_or(0.0);
		}

		let is_inverse = (cluster_byte & SignInverseBit as u8) != 0;
		let cluster_idx = (cluster_byte & !(SignInverseBit as u8)) as usize;

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

	pub fn decompress_all(&self) -> Vec<f32> {
		(0..self.count).map(|i| self.decompress_one(i)).collect()
	}

	pub fn compressed_bytes(&self) -> usize {
		self.packed.len()                                    // packed indices
        + self.prefixes.len() * 4                            // prefix table
        + self.tail_tables.iter().map(|t| t.len() * 4).sum::<usize>()  // tail tables
        + self.uncompressed.len() * 4 // uncompressed fallback
	}

	pub fn original_bytes(&self) -> usize {
		self.count * 4
	}

	pub fn comp_ratio(&self) -> f32 {
		let orig = self.original_bytes() as f32;
		let comp = self.compressed_bytes() as f32;
		if comp == 0.0 { 1.0 } else { orig / comp }
	}
}

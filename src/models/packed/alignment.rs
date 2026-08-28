use super::tensor::PackedTensor;

impl PackedTensor {
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

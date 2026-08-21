use super::tensor::CountIndexedTensor;

impl CountIndexedTensor {
    
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

}
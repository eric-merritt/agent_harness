use crate::models::dedup::types::Sandbag;


impl Sandbag {
    /// Serializes the sandbag structure into a packed binary payload.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.count as u32).to_le_bytes());
        bytes.push(self.tail_width);
        
        // Write out prefix index arrays
        bytes.extend_from_slice(&(self.prefix_idx.len() as u32).to_le_bytes());
        for &p in &self.prefix_idx {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        
        // Write out tail index arrays
        bytes.extend_from_slice(&(self.tail_idx.len() as u32).to_le_bytes());
        for &t in &self.tail_idx {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        
        // Append bitpacked sign streams
        bytes.extend_from_slice(&(self.sign_bits.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.sign_bits);
        
        bytes
    }

    /// Deserializes a packed sandbag instance from a byte array slice.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 9 { return None; }
        let mut pos = 0;
        
        let count = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        let tail_width = data[pos];
        pos += 1;
        
        let p_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        if data.len() < pos + (p_len * 2) { return None; }
        let mut prefix_idx = Vec::with_capacity(p_len);
        for _ in 0..p_len {
            prefix_idx.push(u16::from_le_bytes(data[pos..pos+2].try_into().ok()?));
            pos += 2;
        }
        
        let t_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        if data.len() < pos + (t_len * 2) { return None; }
        let mut tail_idx = Vec::with_capacity(t_len);
        for _ in 0..t_len {
            tail_idx.push(u16::from_le_bytes(data[pos..pos+2].try_into().ok()?));
            pos += 2;
        }
        
        let s_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        if data.len() < pos + s_len { return None; }
        let sign_bits = data[pos..pos+s_len].to_vec();
        
        Some(Self {
            prefix_idx,
            tail_idx,
            tail_width,
            sign_bits,
            count,
        })
    }

    /// Resolves the true total byte size of this serialized segment.
    pub fn bytes(&self) -> usize {
        // count(4) + width(1) + structures lengths headers + payloads
        4 + 1 + 4 + (self.prefix_idx.len() * 2) + 4 + (self.tail_idx.len() * 2) + 4 + self.sign_bits.len()
    }
}
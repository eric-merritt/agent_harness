// Core dictionary binary format: serialize/deserialize DedupCountTensor.
//
// The bidirectional layout:
//   Front: prefix values, then tail values.
//   Back:  tail counts (reversed), then prefix counts (reversed).
// Flags separate the sections. One instance of each unique value — no repeats.

use super::dedup_count::{DedupCountTensor, DataFlag, UniqueTail};

/// Serialize the core dictionary using the bidirectional layout.
pub fn serialize_core(tensor: &DedupCountTensor) -> Vec<u8> {
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(&(tensor.count as u32).to_le_bytes());
    data.extend_from_slice(&(tensor.prefixes.len() as u32).to_le_bytes());
    data.extend_from_slice(&(tensor.unique_tails.len() as u32).to_le_bytes());
    data.extend_from_slice(&(tensor.prefix_digits as u32).to_le_bytes());
    data.extend_from_slice(&(tensor.tail_digits as u32).to_le_bytes());
    data.extend_from_slice(&tensor.avg_precision_lost.to_le_bytes());

    // Front: prefix values (u16 integers, one instance each)
    for &p in &tensor.prefixes {
        data.extend_from_slice(&p.to_le_bytes());
    }
    // Front: tail values (one instance each)
    for ut in &tensor.unique_tails {
        data.extend_from_slice(&ut.value.to_le_bytes());
    }
    data.push(DataFlag::GapFlag as u8);
    // Back: tail counts (reversed — last tail's count first)
    for ut in tensor.unique_tails.iter().rev() {
        data.extend_from_slice(&ut.repeat_count.to_le_bytes());
    }
    data.push(DataFlag::TailFlag as u8);
    // Back: prefix counts (reversed — last prefix's count first)
    for &pc in tensor.prefix_counts.iter().rev() {
        data.extend_from_slice(&pc.to_le_bytes());
    }
    data.push(DataFlag::CountFlag as u8);

    data
}

/// Deserialize the core dictionary from bytes (single chunk, no prefix).
pub fn deserialize_core(data: &[u8]) -> Option<DedupCountTensor> {
    let mut pos = 0;
    deserialize_core_at(data, &mut pos)
}

/// Deserialize a DedupCountTensor starting at `pos`, advancing pos past the data.
pub fn deserialize_core_at(data: &[u8], pos: &mut usize) -> Option<DedupCountTensor> {
    if data.len() < *pos + 24 { return None; }
    let count = u32::from_le_bytes(data[*pos..*pos+4].try_into().ok()?) as usize;
    let prefix_count = u32::from_le_bytes(data[*pos+4..*pos+8].try_into().ok()?) as usize;
    let tail_count = u32::from_le_bytes(data[*pos+8..*pos+12].try_into().ok()?) as usize;
    let prefix_digits = u32::from_le_bytes(data[*pos+12..*pos+16].try_into().ok()?) as usize;
    let tail_digits = u32::from_le_bytes(data[*pos+16..*pos+20].try_into().ok()?) as usize;
    let avg_precision_lost = f32::from_le_bytes(data[*pos+20..*pos+24].try_into().ok()?);
    *pos += 24;

    let mut prefixes = Vec::with_capacity(prefix_count);
    for _ in 0..prefix_count {
        if data.len() < *pos + 2 { return None; }
        prefixes.push(u16::from_le_bytes(data[*pos..*pos+2].try_into().ok()?));
        *pos += 2;
    }

    let mut unique_tails = Vec::with_capacity(tail_count);
    for _ in 0..tail_count {
        if data.len() < *pos + 2 { return None; }
        let value = u16::from_le_bytes(data[*pos..*pos+2].try_into().ok()?);
        unique_tails.push(UniqueTail { value, repeat_count: 0 });
        *pos += 2;
    }

    if *pos < data.len() && data[*pos] == DataFlag::GapFlag as u8 { *pos += 1; }

    let tail_counts_bytes = tail_count * 4;
    let prefix_counts_bytes = prefix_count * 4;
    let total_rear_bytes = tail_counts_bytes + 1 + prefix_counts_bytes + 1;
    if data.len() < *pos + total_rear_bytes { return None; }

    let tail_counts_start = *pos;
    let tail_flag_pos = tail_counts_start + tail_counts_bytes;
    let prefix_counts_start = tail_flag_pos + 1;
    let count_flag_pos = prefix_counts_start + prefix_counts_bytes;

    if data[tail_flag_pos] != DataFlag::TailFlag as u8 { return None; }
    if data[count_flag_pos] != DataFlag::CountFlag as u8 { return None; }

    for (i, ut) in unique_tails.iter_mut().rev().enumerate() {
        let chunk_offset = tail_counts_start + (i * 4);
        ut.repeat_count = u32::from_le_bytes(data[chunk_offset..chunk_offset + 4].try_into().ok()?);
    }

    let mut prefix_counts = vec![0u32; prefix_count];
    for i in 0..prefix_count {
        let chunk_offset = prefix_counts_start + (i * 4);
        prefix_counts[prefix_count - 1 - i] = u32::from_le_bytes(data[chunk_offset..chunk_offset + 4].try_into().ok()?);
    }

    *pos = count_flag_pos + 1;

    Some(DedupCountTensor {
        count,
        prefixes,
        unique_tails,
        prefix_counts,
        prefix_digits,
        tail_digits,
        avg_precision_lost,
    })
}

/// Deserialize all chunks from a chunked core (chunk_count prefix + N tensors).
pub fn deserialize_core_chunks(data: &[u8]) -> Vec<DedupCountTensor> {
    if data.len() < 4 { return Vec::new(); }
    let chunk_count = u32::from_le_bytes(data[0..4].try_into().unwrap_or([0,0,0,0])) as usize;
    let mut pos = 4;
    let mut chunks = Vec::with_capacity(chunk_count);
    for _ in 0..chunk_count {
        if let Some(t) = deserialize_core_at(data, &mut pos) {
            chunks.push(t);
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_serialize_roundtrip() {
        let weights: Vec<f32> = (0..1000)
            .map(|i| match i % 3 { 0 => 0.0150, 1 => -0.0230, _ => 0.0420 })
            .collect();

        let (tensor, _sandbag) = DedupCountTensor::compress(&weights, 2, 2);
        let bytes = serialize_core(&tensor);
        let restored = deserialize_core(&bytes).expect("deserialization failed");

        assert_eq!(restored.count, tensor.count);
        assert_eq!(restored.prefixes.len(), tensor.prefixes.len());
        assert_eq!(restored.unique_tails.len(), tensor.unique_tails.len());
        assert_eq!(restored.prefix_digits, tensor.prefix_digits);
        assert_eq!(restored.tail_digits, tensor.tail_digits);
        assert!((restored.avg_precision_lost - tensor.avg_precision_lost).abs() < 1e-6);

        for (a, b) in tensor.prefixes.iter().zip(restored.prefixes.iter()) {
            assert_eq!(a, b);
        }
        for (a, b) in tensor.unique_tails.iter().zip(restored.unique_tails.iter()) {
            assert_eq!(a.value, b.value);
            assert_eq!(a.repeat_count, b.repeat_count);
        }
    }
}

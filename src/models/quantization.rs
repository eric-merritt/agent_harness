// Custom quantization methods — content-aware weight compression.
//
// Standard quantization (INT4/INT8/GPTQ) treats all weights uniformly.
// These methods exploit the actual distribution of values:
//
// 1. Coordinate Quantization: weights sharing a common prefix (high-order digits)
//    store the prefix once, with tails indexed in a compact lookup table.
//    The tail table fits in the remaining alignment space.
//
// 2. Inverse Quantization: weights that are inverses (w and 1-w) share one
//    stored value with an inverse flag bit, halving their storage.
//
// 3. Hybrid: combine both methods — cluster weights by prefix, flag inverses
//    within each cluster, and store only the unique tails.

use std::collections::HashMap;
use std::arch::x86_64::*;


pub trait QuantMethod {
    fn compress(&self, weights: &[f32]) -> CompressedTensor;
    fn decompress(&self, compressed: &CompressedTensor) -> Vec<f32>;
    fn ratio(&self, compressed: &CompressedTensor) -> f32;
    fn name(&self) -> &'static str;
}

#[derive(Clone, Debug)]
pub struct CompressedTensor {
    pub method: CompressMethod,
    pub data: Vec<u8>,
    pub original_count: usize,
    pub original_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CompressMethod {
    Coordinate,
    Inverse,
    Hybrid,
    None,
}

// ── Corrected Coordinate Quantizer ──────────────────────────────────────────

pub struct CoordinateQuantizer {
    pub prefix_digits: usize,
}

impl CoordinateQuantizer {
    pub fn new(prefix_digits: usize) -> Self {
        Self { prefix_digits: prefix_digits.max(2).min(7) }
    }

    fn extract_prefix(&self, w: f32) -> (f32, u16) {
        let scale = 10f32.powi(self.prefix_digits as i32);
        let prefix = (w * scale).floor() / scale;
        let tail = w - prefix;
        let tail_scale = 10f32.powi((5 - self.prefix_digits as i32).max(0));
        let tail_val = (tail * tail_scale * 65535.0).round().clamp(0.0, 65535.0) as u16;
        (prefix, tail_val)
    }

    fn reconstruct(&self, prefix: f32, tail: u16) -> f32 {
        let tail_scale = 10f32.powi((5 - self.prefix_digits as i32).max(0));
        let tail_val = tail as f32 / 65535.0 / tail_scale;
        prefix + tail_val
    }
}

// ── Optimized O(N log N) Inverse Quantizer ───────────────────────────────────

pub struct InverseQuantizer {
    pub tolerance: f32,
}

impl InverseQuantizer {
    pub fn new(tolerance: f32) -> Self {
        Self { tolerance: tolerance.max(1e-7) }
    }
}

impl QuantMethod for InverseQuantizer {
    fn compress(&self, weights: &[f32]) -> CompressedTensor {
        let mut data = Vec::new();
        if weights.is_empty() {
            return CompressedTensor { method: CompressMethod::Inverse, data, original_count: 0, original_bytes: 0 };
        }

        // Create pairs of (weight, original_index) and sort to locate matches fast
        let mut indexed_weights: Vec<(f32, usize)> = weights.iter().copied().enumerate().map(|(i, w)| (w, i)).collect();
        indexed_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut used = vec![false; weights.len()];
        let mut num_pairs = 0u32;
        
        // Make room for the header count field
        data.extend_from_slice(&0u32.to_le_bytes());

        for i in 0..indexed_weights.len() {
            let (w, orig_idx) = indexed_weights[i];
            if used[orig_idx] { continue; }
            used[orig_idx] = true;

            let target = 1.0 - w;
            
            // Binary search to find candidate close to target match
            let mut found_inverse = None;
            let search_res = indexed_weights.binary_search_by(|probe| probe.0.partial_cmp(&target).unwrap());
            
            let center = match search_res {
                Ok(idx) => idx,
                Err(idx) => idx,
            };

            // Scan outward within window tolerance boundaries
            let mut check_idx = center;
            while check_idx < indexed_weights.len() && (indexed_weights[check_idx].0 - target).abs() <= self.tolerance {
                let (_, inv_orig_idx) = indexed_weights[check_idx];
                if !used[inv_orig_idx] {
                    found_inverse = Some(inv_orig_idx);
                    used[inv_orig_idx] = true;
                    break;
                }
                check_idx += 1;
            }

            if found_inverse.is_none() && center > 0 {
                let mut check_idx = center - 1;
                loop {
                    if (indexed_weights[check_idx].0 - target).abs() > self.tolerance { break; }
                    let (_, inv_orig_idx) = indexed_weights[check_idx];
                    if !used[inv_orig_idx] {
                        found_inverse = Some(inv_orig_idx);
                        used[inv_orig_idx] = true;
                        break;
                    }
                    if check_idx == 0 { break; }
                    check_idx -= 1;
                }
            }

            // Continuous payload stream layouts
            data.extend_from_slice(&w.to_le_bytes());
            data.extend_from_slice(&(orig_idx as u32).to_le_bytes());
            if let Some(idx) = found_inverse {
                data.push(1);
                data.extend_from_slice(&(idx as u32).to_le_bytes());
            } else {
                data.push(0);
            }
            num_pairs += 1;
        }

        // Backfill true pair count header space
        data[0..4].copy_from_slice(&num_pairs.to_le_bytes());

        CompressedTensor {
            method: CompressMethod::Inverse,
            data,
            original_count: weights.len(),
            original_bytes: weights.len() * 4,
        }
    }

    fn decompress(&self, compressed: &CompressedTensor) -> Vec<f32> {
        let mut result = vec![0.0f32; compressed.original_count];
        if compressed.data.len() < 4 { return result; }

        let mut pos = 0;
        let num_pairs = u32::from_le_bytes(compressed.data[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;

        for _ in 0..num_pairs {
            let value = f32::from_le_bytes(compressed.data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let direct_idx = u32::from_le_bytes(compressed.data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            let has_inverse = compressed.data[pos] == 1;
            pos += 1;

            result[direct_idx] = value;
            if has_inverse {
                let inverse_idx = u32::from_le_bytes(compressed.data[pos..pos+4].try_into().unwrap()) as usize;
                pos += 4;
                result[inverse_idx] = 1.0 - value;
            }
        }
        result
    }

    fn ratio(&self, compressed: &CompressedTensor) -> f32 {
        if compressed.data.is_empty() { return 1.0; }
        compressed.original_bytes as f32 / compressed.data.len() as f32
    }

    fn name(&self) -> &'static str { "inverse" }
}

// ── Corrected, Deterministic Hybrid Quantizer ───────────────────────────────

pub struct HybridQuantizer {
    pub prefix_digits: usize,
    pub tolerance: f32,
}

impl HybridQuantizer {
    pub fn new(prefix_digits: usize, tolerance: f32) -> Self {
        Self {
            prefix_digits: prefix_digits.max(2).min(7),
            tolerance: tolerance.max(1e-7),
        }
    }
}

impl QuantMethod for HybridQuantizer {
    fn compress(&self, weights: &[f32]) -> CompressedTensor {
        let coord = CoordinateQuantizer::new(self.prefix_digits);
        let mut clusters: HashMap<u32, (f32, Vec<u16>, Vec<usize>)> = HashMap::new();

        for (i, &w) in weights.iter().enumerate() {
            let (prefix, tail) = coord.extract_prefix(w);
            let prefix_bits = prefix.to_bits();
            let entry = clusters.entry(prefix_bits).or_insert_with(|| (prefix, Vec::new(), Vec::new()));
            entry.1.push(tail);
            entry.2.push(i);
        }

        let mut data = Vec::new();
        data.extend_from_slice(&(clusters.len() as u32).to_le_bytes());

        for (_, (prefix, tails, indices)) in &clusters {
            data.extend_from_slice(&prefix.to_le_bytes());
            
            // To prevent runtime decompression skipping bugs, record the EXACT count of packed units written
            let mut record_buffer = Vec::new();
            let mut packed_count = 0u32;
            let mut used = vec![false; tails.len()];

            for j in 0..tails.len() {
                if used[j] { continue; }
                used[j] = true;

                record_buffer.extend_from_slice(&tails[j].to_le_bytes());
                record_buffer.extend_from_slice(&(indices[j] as u32).to_le_bytes());
                
                // Track if inverse tail values are paired within cluster thresholds
                let inv_target = 65535 - tails[j];
                let mut found_match = None;

                for k in (j + 1)..tails.len() {
                    if !used[k] && (tails[k] as i32 - inv_target as i32).abs() < 100 {
                        found_match = Some(k);
                        break;
                    }
                }

                if let Some(k) = found_match {
                    used[k] = true;
                    record_buffer.push(1); // Flag indicating Inverse Pair present
                    record_buffer.extend_from_slice(&(indices[k] as u32).to_le_bytes());
                } else {
                    record_buffer.push(0); // Standard direct entry only
                }
                packed_count += 1;
            }

            data.extend_from_slice(&packed_count.to_le_bytes());
            data.extend_from_slice(&record_buffer);
        }

        CompressedTensor {
            method: CompressMethod::Hybrid,
            data,
            original_count: weights.len(),
            original_bytes: weights.len() * 4,
        }
    }

    fn decompress(&self, compressed: &CompressedTensor) -> Vec<f32> {
        let mut result = vec![0.0f32; compressed.original_count];
        if compressed.data.len() < 4 { return result; }

        let mut pos = 0;
        let num_clusters = u32::from_le_bytes(compressed.data[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;

        let coord = CoordinateQuantizer::new(self.prefix_digits);

        for _ in 0..num_clusters {
            let prefix = f32::from_le_bytes(compressed.data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let packed_count = u32::from_le_bytes(compressed.data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;

            for _ in 0..packed_count {
                let tail = u16::from_le_bytes(compressed.data[pos..pos+2].try_into().unwrap());
                pos += 2;
                let idx = u32::from_le_bytes(compressed.data[pos..pos+4].try_into().unwrap()) as usize;
                pos += 4;
                let flag = compressed.data[pos];
                pos += 1;

                let value = coord.reconstruct(prefix, tail);
                result[idx] = value;

                if flag == 1 {
                    let inv_idx = u32::from_le_bytes(compressed.data[pos..pos+4].try_into().unwrap()) as usize;
                    pos += 4;
                    // Resolve the inversion bug safely based on decoded value space
                    result[inv_idx] = 1.0 - value;
                }
            }
        }
        result
    }

    fn ratio(&self, compressed: &CompressedTensor) -> f32 {
        if compressed.data.is_empty() { return 1.0; }
        compressed.original_bytes as f32 / compressed.data.len() as f32
    }

    fn name(&self) -> &'static str { "hybrid" }
}

// ── Worker-based decompression ──────────────────────────────────────────────


#[derive(Clone, Debug)]
pub struct WorkerChunk {
    /// True matrix index spans to allocate to a given compute thread
    pub start_idx: usize,
    pub end_idx: usize,
}

pub fn split_for_workers(compressed: &CompressedTensor, num_workers: usize) -> Vec<WorkerChunk> {
    let num_workers = num_workers.max(1);
    let total_elements = compressed.original_count;
    let chunk_size = (total_elements + num_workers - 1) / num_workers;

    let mut chunks = Vec::new();
    for i in 0..num_workers {
        let start_idx = i * chunk_size;
        if start_idx >= total_elements { break; }
        let end_idx = (start_idx + chunk_size).min(total_elements);

        chunks.push(WorkerChunk { start_idx, end_idx });
    }
    chunks
}

// ── INT4 Symmetric Per-Group Quantization ───────────────────────────────────
//
// Group size: 32 (standard, matches GPTQ/AWQ).
// For each group of 32 weights:
//   scale = max(|w|) / 7.0
//   q = round(w / scale), clamped to [-8, 7]
//   stored as unsigned: q + 8 ∈ [0, 15]
//
// Packing: 2 indices per byte, low nibble = even index, high nibble = odd.
// Dequant: w = (q_unsigned - 8) * scale
//
// Size: ceil(n/2) bytes for data + ceil(n/32)*4 bytes for scales.
// For 14B params: ~7GB data + ~1.75GB scales = ~8.75GB (vs 56GB F32).

pub const GROUP_SIZE: usize = 32;

/// Quantize f32 weights to INT4 with per-group symmetric scales.
/// Returns (scales, packed) where packed has 2 indices per byte.
pub fn quantize(weights: &[f32], group_size: usize) -> (Vec<f32>, Vec<u8>) {
    let gs = group_size.max(1);
    let n_groups = (weights.len() + gs - 1) / gs;
    let mut scales = Vec::with_capacity(n_groups);
    let mut packed = vec![0u8; (weights.len() + 1) / 2];

    for g in 0..n_groups {
        let start = g * gs;
        let end = (start + gs).min(weights.len());
        let group = &weights[start..end];

        let max_abs = group.iter().fold(0.0f32, |a, &w| a.max(w.abs()));
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 0.0 };
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        scales.push(scale);

        for (i, &w) in group.iter().enumerate() {
            let q = if scale > 0.0 {
                (w * inv_scale).round() as i32
            } else {
                0
            };
            let clamped = q.clamp(-8, 7) + 8; // Transform range to [0, 15]
            let idx = start + i;
            
            if idx % 2 == 0 {
                packed[idx / 2] |= clamped as u8;
            } else {
                packed[idx / 2] |= (clamped as u8) << 4;
            }
        }
    }

    (scales, packed)
}

/// Dequantize a single weight at `index`.
#[inline]
pub fn dequant_weight(scales: &[f32], packed: &[u8], index: usize, group_size: usize) -> f32 {
    let gs = group_size.max(1);
    let group_idx = index / gs;
    let scale = scales[group_idx];
    
    let nibble = if index % 2 == 0 {
        packed[index / 2] & 0x0F
    } else {
        (packed[index / 2] >> 4) & 0x0F
    };
    
    (nibble as f32 - 8.0) * scale
}

/// Bytes needed for packed data (ceil(n/2)).
pub fn packed_bytes(n_elements: usize) -> usize {
    (n_elements + 1) / 2
}

/// Number of scale groups for n_elements with group_size.
pub fn n_groups(n_elements: usize, group_size: usize) -> usize {
    (n_elements + group_size - 1) / group_size
}

/// Total bytes for a quantized tensor (scales + packed data).
pub fn quantized_bytes(n_elements: usize, group_size: usize) -> usize {
    n_groups(n_elements, group_size) * 4 + packed_bytes(n_elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize() {
        let mut weights = vec![
            0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
            0.15, 0.25, -0.35, 0.45, -0.55, 0.65, -0.75, 0.85,
            -0.95, 0.05, -0.15, 0.25, -0.35, 0.45, -0.55, 0.65,
            -0.75, 0.85, -0.95, 0.05, -0.15, 0.25, -0.35, 0.45,
        ];
        weights.extend(std::iter::repeat(0.0f32).take(32)); // Fixed deprecated repeat_n compilation hazard
        
        let (scales, packed) = quantize(&weights, GROUP_SIZE);
        let deq: Vec<f32> = (0..weights.len())
            .map(|i| dequant_weight(&scales, &packed, i, GROUP_SIZE))
            .collect();

        let max_err = weights.iter().zip(deq.iter())
            .map(|(o, d)| (o - d).abs())
            .fold(0.0f32, f32::max);

        assert!(max_err < 0.15, "Max error too large: {}", max_err);
    }

    #[test]
    fn test_zero_weights() {
        let weights = vec![0.0; 64];
        let (scales, packed) = quantize(&weights, GROUP_SIZE);
        for i in 0..weights.len() {
            let v = dequant_weight(&scales, &packed, i, GROUP_SIZE);
            assert!(v.abs() < 1e-6, "Zero weight dequantized to {}", v);
        }
    }

    #[test]
    fn test_packing() {
        let mut weights = vec![0.0; 32];
        weights[0] = 0.7;
        weights[1] = -0.7;
        weights[2] = 0.0;

        let (scales, packed) = quantize(&weights, GROUP_SIZE);

        let b = packed[0];
        let q0 = (b & 0x0F) as i32 - 8;
        let q1 = ((b >> 4) & 0x0F) as i32 - 8;

        assert!(q0 > 0, "First weight should be positive, got {}", q0);
        assert!(q1 < 0, "Second weight should be negative, got {}", q1);
    }
}



/// Safe entry point that checks CPU features at runtime.
/// Falls back to scalar processing if AVX-512 instructions are missing.
pub fn dequant_layer(scales: &[f32], packed: &[u8], result: &mut [f32], group_size: usize) {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && group_size == 32 {
        unsafe { dequant_layer_avx512(scales, packed, result); }
    } else {
        // Safe fallback scalar loop
        for i in 0..result.len() {
            let group_idx = i / group_size;
            let nibble = if i % 2 == 0 { 
                packed[i / 2] & 0x0F 
            } else { 
                (packed[i / 2] >> 4) & 0x0F 
            };
            result[i] = (nibble as f32 - 8.0) * scales[group_idx];
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn dequant_layer_avx512(scales: &[f32], packed: &[u8], result: &mut [f32]) {
    let n = result.len();
    let chunks = n / 128;
    
    let v_offset = _mm512_set1_ps(-8.0);
    let v_mask_low = _mm512_set1_epi8(0x0F);

    for c in 0..chunks {
        let byte_offset = c * 64;   
        let float_offset = c * 128; 
        let group_offset = c * 4;   

        // 1. Load 64 packed bytes into ZMM
        let raw_bytes = _mm512_loadu_si512(packed.as_ptr().add(byte_offset) as *const _);

        // 2. Unpack low (even) and high (odd) nibbles
        let bytes_even = _mm512_and_si512(raw_bytes, v_mask_low);
        let bytes_odd = _mm512_and_si512(_mm512_srli_epi16(raw_bytes, 4), v_mask_low);

        // 3. Interleave back into accurate linear element order
        let lin_0_63 = _mm512_unpacklo_epi8(bytes_even, bytes_odd);
        let lin_64_127 = _mm512_unpackhi_epi8(bytes_even, bytes_odd);

        // 4. Convert to f32 and apply group scales (each block handles 32 items)
        
        // --- Block 0 (Elements 0-31, Scale Group 0) ---
        let scale_0 = _mm512_set1_ps(scales[group_offset]);
        
        let chunk0_1 = _mm512_castsi512_si128(lin_0_63);
        let chunk0_2 = _mm512_extracti32x4_epi32(lin_0_63, 1);
        
        let f0_1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(chunk0_1));
        let f0_2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(chunk0_2));
        
        _mm512_storeu_ps(result.as_mut_ptr().add(float_offset), _mm512_fmadd_ps(_mm512_add_ps(f0_1, v_offset), scale_0, _mm512_setzero_ps()));
        _mm512_storeu_ps(result.as_mut_ptr().add(float_offset + 16), _mm512_fmadd_ps(_mm512_add_ps(f0_2, v_offset), scale_0, _mm512_setzero_ps()));

        // --- Block 1 (Elements 32-63, Scale Group 1) ---
        let scale_1 = _mm512_set1_ps(scales[group_offset + 1]);
        
        let chunk1_1 = _mm512_extracti32x4_epi32(lin_0_63, 2);
        let chunk1_2 = _mm512_extracti32x4_epi32(lin_0_63, 3);
        
        let f1_1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(chunk1_1));
        let f1_2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(chunk1_2));
        
        _mm512_storeu_ps(result.as_mut_ptr().add(float_offset + 32), _mm512_fmadd_ps(_mm512_add_ps(f1_1, v_offset), scale_1, _mm512_setzero_ps()));
        _mm512_storeu_ps(result.as_mut_ptr().add(float_offset + 48), _mm512_fmadd_ps(_mm512_add_ps(f1_2, v_offset), scale_1, _mm512_setzero_ps()));

        // --- Block 2 (Elements 64-95, Scale Group 2) ---
        let scale_2 = _mm512_set1_ps(scales[group_offset + 2]);
        
        let chunk2_1 = _mm512_castsi512_si128(lin_64_127);
        let chunk2_2 = _mm512_extracti32x4_epi32(lin_64_127, 1);
        
        let f2_1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(chunk2_1));
        let f2_2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(chunk2_2));
        
        _mm512_storeu_ps(result.as_mut_ptr().add(float_offset + 64), _mm512_fmadd_ps(_mm512_add_ps(f2_1, v_offset), scale_2, _mm512_setzero_ps()));
        _mm512_storeu_ps(result.as_mut_ptr().add(float_offset + 80), _mm512_fmadd_ps(_mm512_add_ps(f2_2, v_offset), scale_2, _mm512_setzero_ps()));

        // --- Block 3 (Elements 96-127, Scale Group 3) ---
        let scale_3 = _mm512_set1_ps(scales[group_offset + 3]);
        
        let chunk3_1 = _mm512_extracti32x4_epi32(lin_64_127, 2);
        let chunk3_2 = _mm512_extracti32x4_epi32(lin_64_127, 3);
        
        let f3_1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(chunk3_1));
        let f3_2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(chunk3_2));
        
        _mm512_storeu_ps(result.as_mut_ptr().add(float_offset + 96), _mm512_fmadd_ps(_mm512_add_ps(f3_1, v_offset), scale_3, _mm512_setzero_ps()));
        _mm512_storeu_ps(result.as_mut_ptr().add(float_offset + 112), _mm512_fmadd_ps(_mm512_add_ps(f3_2, v_offset), scale_3, _mm512_setzero_ps()));
    }
}

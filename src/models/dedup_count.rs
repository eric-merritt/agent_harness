// DedupCountTensor compression.
//
// The core weights file stores ONLY unique values:
//   - One instance of each prefix (e.g., .25, .73, .47, .29 — no repeats)
//   - One instance of each tail (e.g., 26, 37, 42, 51 — no repeats)
//   - A verification count (how many tails to recover)
//
// The sandbag file stores the per-weight map (which prefix, which tail, sign).
// It stays on disk — never loaded as data. A worker uses it to design the
// unified memory space, wrapping weights into grids for zerocopy into the server.
//
// Compression ratio = core_bytes / original_bytes.
// The sandbag is not counted — it's a disk-based index, not loaded data.

use ahash::AHashMap;

/// Shared global dictionary of unique prefix→tail mappings across ALL chunks/tensors.
/// This is the "zero copy surface" — loaded once, shared via Arc, workers only dereference indices.
///
/// Layout: for each unique prefix value, a list of unique tail values that co-occur with it.
/// Sandbag indices: prefix_idx → index into prefixes, tail_idx → index into tails_for_prefix[prefix_idx].
///
/// CSR (flat) layout is also available for GPU consumption and direct indexing:
///   flat_tails[tail_offsets[prefix_idx] + tail_idx] gives the tail value.
#[derive(Clone, Debug)]
pub struct GlobalTable {
    pub prefixes: Vec<f32>,                       // all unique prefix values, stored as idx / 10^prefix_digits
    pub tails_for_prefix: Vec<Vec<u32>>,           // per-prefix unique tail values
    // CSR layout — flat for O(1) direct indexing, GPU-friendly
    pub flat_tails: Vec<u32>,                      // all tails concatenated
    pub tail_offsets: Vec<u32>,                     // CSR row pointers: tail_offsets[p] = start in flat_tails
    pub prefix_digits: usize,                       // scale used for normalization (max across all tensors)
}

impl GlobalTable {
    pub fn new() -> Self {
        Self {
            prefixes: Vec::new(),
            tails_for_prefix: Vec::new(),
            flat_tails: Vec::new(),
            tail_offsets: Vec::new(),
            prefix_digits: 2,
        }
    }

    /// Build from multiple DedupCountTensor chunks. Collects all unique prefix→tail mappings
    /// from the tensor data only (doesn't need the sandbag). Uses the maximum prefix_digits
    /// across all tensors as the normalization scale — this prevents collisions between
    /// 2-digit and 4-digit prefixes.
    pub fn from_tensors(tensors: &[&DedupCountTensor]) -> Self {
        let max_pd = tensors.iter().map(|t| t.prefix_digits).max().unwrap_or(2);
        let mut prefix_map: AHashMap<u16, AHashMap<u32, ()>> = AHashMap::new();
        for tensor in tensors {
            for &prefix_int in &tensor.prefixes {
                let entry = prefix_map.entry(prefix_int).or_default();
                for ut in &tensor.unique_tails {
                    entry.insert(ut.value as u32, ());
                }
            }
        }
        let mut prefix_indices: Vec<u16> = prefix_map.keys().copied().collect();
        prefix_indices.sort();
        let scale = 10f32.powi(max_pd as i32);
        let prefixes: Vec<f32> = prefix_indices.iter().map(|&i| i as f32 / scale).collect();
        let tails_for_prefix: Vec<Vec<u32>> = prefix_indices.iter()
            .map(|&idx| {
                let mut tails: Vec<u32> = prefix_map[&idx].keys().copied().collect();
                tails.sort();
                tails
            })
            .collect();

        // Build CSR layout
        let mut flat_tails = Vec::new();
        let mut tail_offsets = Vec::with_capacity(prefixes.len() + 1);
        for tails in &tails_for_prefix {
            tail_offsets.push(flat_tails.len() as u32);
            flat_tails.extend_from_slice(tails);
        }
        tail_offsets.push(flat_tails.len() as u32); // sentinel

        Self { prefixes, tails_for_prefix, flat_tails, tail_offsets, prefix_digits: max_pd }
    }

    /// Build a per-chunk remap table that translates chunk-local sandbag indices
    /// to global indices. This eliminates all HashMap construction in the hot path.
    /// Call once per chunk at load time; workers then do pure direct indexing.
    pub fn build_chunk_remap(&self, chunk: &DedupCountTensor) -> ChunkRemap {
        // prefix_remap: chunk_prefix_idx → global_prefix_idx
        // Convert chunk prefix (chunk.prefix_digits scale) to GlobalTable scale (self.prefix_digits).
        // e.g., chunk prefix 25 with 2-digit → GlobalTable 2500 with 4-digit
        let gt_scale = 10f32.powi(self.prefix_digits as i32);
        let scale_diff = self.prefix_digits as i32 - chunk.prefix_digits as i32;
        let mut prefix_remap = Vec::with_capacity(chunk.prefixes.len());
        for &pv in &chunk.prefixes {
            let norm = if scale_diff > 0 {
                (pv as u32) * 10u32.pow(scale_diff as u32)
            } else {
                pv as u32
            };
            let gidx = self.prefixes.iter().position(|&gp| (gp * gt_scale).round() as u32 == norm)
                .unwrap_or(0);
            prefix_remap.push(gidx as u16);
        }

        // tail_remap_flat: for each chunk prefix, maps chunk_tail_idx → global_tail_idx
        let mut tail_remap_flat = Vec::new();
        let mut tail_remap_offsets = Vec::with_capacity(chunk.prefixes.len() + 1);
        for (cp_idx, &_pv) in chunk.prefixes.iter().enumerate() {
            tail_remap_offsets.push(tail_remap_flat.len() as u16);
            let gp_idx = prefix_remap[cp_idx] as usize;
            let global_tails = &self.tails_for_prefix[gp_idx];
            let mut val_to_gi: AHashMap<u32, u16> = AHashMap::with_capacity(global_tails.len());
            for (gi, &t) in global_tails.iter().enumerate() {
                val_to_gi.insert(t, gi as u16);
            }
            for ut in &chunk.unique_tails {
                let gi = val_to_gi.get(&(ut.value as u32)).copied().unwrap_or(0);
                tail_remap_flat.push(gi);
            }
        }
        tail_remap_offsets.push(tail_remap_flat.len() as u16);

        ChunkRemap { prefix_remap, tail_remap_flat, tail_remap_offsets }
    }

    /// Fast decompression using CSR layout + precomputed remap.
    /// Pure direct indexing — no HashMap construction.
    pub fn decompress_with_remap(
        &self,
        sandbag: &Sandbag,
        chunk: &DedupCountTensor,
        remap: &ChunkRemap,
    ) -> Vec<f32> {
        // GlobalTable stores prefixes as f32 (i/10^prefix_digits) and tails as raw u32 integers.
        // The tail divisor must use the GLOBAL table's digit counts, not the chunk's,
        // because the tail values come from the global table's flat_tails.
        let tail_divisor = 10f32.powi((self.prefix_digits + chunk.tail_digits) as i32);
        let avg_pl = chunk.avg_precision_lost;
        let mut result = Vec::with_capacity(chunk.count);

        for i in 0..chunk.count {
            let cp_idx = sandbag.prefix_idx.get(i).copied().unwrap_or(0) as usize;
            let ct_idx = sandbag.tail_idx.get(i).copied().unwrap_or(0) as usize;

            let gp = remap.prefix_remap[cp_idx] as usize;
            let gt_base = remap.tail_remap_offsets[cp_idx] as usize;
            let gt = remap.tail_remap_flat[gt_base + ct_idx] as usize;

            let prefix = self.prefixes[gp];
            let tail = self.flat_tails[self.tail_offsets[gp] as usize + gt];

            let mut value = prefix + tail as f32 / tail_divisor + avg_pl;
            let sign = (sandbag.sign_bits.get(i / 8).copied().unwrap_or(0) >> (i % 8)) & 1 != 0;
            if sign { value = -value; }
            result.push(value);
        }

        result
    }

    /// Look up the tail index for a given prefix index and tail value.
    pub fn find(&self, prefix: f32, tail: u32) -> Option<(usize, usize)> {
        let scale = 10f32.powi(self.prefix_digits as i32);
        let norm = (prefix * scale).round() as u16;
        let prefix_idx = self.prefixes.iter().position(|&p| (p * scale).round() as u16 == norm)?;
        let tail_idx = self.tails_for_prefix[prefix_idx].iter().position(|&t| t == tail)?;
        Some((prefix_idx, tail_idx))
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&(self.prefix_digits as u32).to_le_bytes());
        data.extend_from_slice(&(self.prefixes.len() as u32).to_le_bytes());
        for &p in &self.prefixes {
            data.extend_from_slice(&p.to_le_bytes());
        }
        // CSR: tail_offsets (prefix_count+1) + flat_tails
        data.extend_from_slice(&(self.tail_offsets.len() as u32).to_le_bytes());
        for &o in &self.tail_offsets {
            data.extend_from_slice(&o.to_le_bytes());
        }
        data.extend_from_slice(&(self.flat_tails.len() as u32).to_le_bytes());
        for &t in &self.flat_tails {
            data.extend_from_slice(&t.to_le_bytes());
        }
        data
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        let mut pos = 0;
        if data.len() < 8 { return None; }
        let prefix_digits = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        pos += 4;
        let prefix_count = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        let mut prefixes = Vec::with_capacity(prefix_count);
        for _ in 0..prefix_count {
            if pos + 4 > data.len() { return None; }
            prefixes.push(f32::from_le_bytes(data[pos..pos+4].try_into().ok()?));
            pos += 4;
        }
        // Read CSR: tail_offsets then flat_tails
        if pos + 4 > data.len() { return None; }
        let offset_count = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        let mut tail_offsets = Vec::with_capacity(offset_count);
        for _ in 0..offset_count {
            if pos + 4 > data.len() { return None; }
            tail_offsets.push(u32::from_le_bytes(data[pos..pos+4].try_into().ok()?));
            pos += 4;
        }
        if pos + 4 > data.len() { return None; }
        let flat_count = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        let mut flat_tails = Vec::with_capacity(flat_count);
        for _ in 0..flat_count {
            if pos + 4 > data.len() { return None; }
            flat_tails.push(u32::from_le_bytes(data[pos..pos+4].try_into().ok()?));
            pos += 4;
        }

        // Reconstruct tails_for_prefix from CSR
        let mut tails_for_prefix = Vec::with_capacity(prefix_count);
        for i in 0..prefix_count {
            let start = tail_offsets[i] as usize;
            let end = tail_offsets.get(i + 1).copied().unwrap_or(flat_tails.len() as u32) as usize;
            tails_for_prefix.push(flat_tails[start..end].to_vec());
        }

        Some(Self { prefixes, tails_for_prefix, flat_tails, tail_offsets, prefix_digits })
    }
}

/// Precomputed per-chunk remap table. Translates chunk-local sandbag indices
/// to global GlobalTable indices. Built once at load time — eliminates all
/// HashMap construction from the decompression hot path.
#[derive(Clone, Debug)]
pub struct ChunkRemap {
    /// chunk_prefix_idx → global_prefix_idx
    pub prefix_remap: Vec<u16>,
    /// Flattened tail remap: for chunk_prefix_idx cp,
    /// tail_remap_flat[tail_remap_offsets[cp] + chunk_tail_idx] → global_tail_idx
    pub tail_remap_flat: Vec<u16>,
    /// CSR offsets into tail_remap_flat
    pub tail_remap_offsets: Vec<u16>,
}

fn prefix_bits(prefix: f32) -> u32 { prefix.to_bits() }

/// Flag markers for the bidirectional section layout.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum DataFlag {
    GapFlag = 0xFD,
    TailFlag = 0xFE,
    CountFlag = 0xFC,
}

/// A unique tail value with its repeat count (for the core's tail section).
#[derive(Clone, Debug)]
pub struct UniqueTail {
    pub value: u16,
    pub repeat_count: u32,
}

/// Per-weight map. Stored in sandbag.bin, stays on disk.
///
/// A worker reads this to set up the memory layout. It maps each weight
/// to its prefix index and tail index in the core dictionary.
///
/// File layout:
///   [count: u32]
///   [tail_width: u8]            — 0 = u8 tail indices, 1 = u16 tail indices
///   [prefix_idx: u8 * N]        — which prefix in the core's prefix list
///   [tail_idx: u8 * N | u16 * N] — which tail in the core's tail list (width depends on tail_width)
///   [sign_bits: u8 * ceil(N/8)]
#[derive(Clone, Debug)]
pub struct Sandbag {
    pub prefix_idx: Vec<u8>,
    pub tail_idx: Vec<u16>,
    pub tail_width: u8,
    pub sign_bits: Vec<u8>,
    pub count: usize,
}

impl Sandbag {
    pub fn to_bytes(&self) -> Vec<u8> {
        let sign_bytes = (self.count + 7) / 8;
        let tail_bytes = self.count * if self.tail_width == 0 { 1 } else { 2 };
        let mut data = Vec::with_capacity(4 + 1 + self.count + tail_bytes + sign_bytes);
        data.extend_from_slice(&(self.count as u32).to_le_bytes());
        data.push(self.tail_width);
        data.extend_from_slice(&self.prefix_idx);
        if self.tail_width == 0 {
            for &v in &self.tail_idx {
                data.push(v as u8);
            }
        } else {
            for &v in &self.tail_idx {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        data.extend_from_slice(&self.sign_bits);
        data
    }

    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 5 { return None; }
        let count = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let tail_width = data[4];
        let prefix_end = 5 + count;
        if data.len() < prefix_end { return None; }
        let prefix_idx = data[5..prefix_end].to_vec();

        let (tail_idx, sign_start) = if tail_width == 0 {
            let tail_end = prefix_end + count;
            if data.len() < tail_end { return None; }
            let ti: Vec<u16> = data[prefix_end..tail_end].iter().map(|&b| b as u16).collect();
            (ti, tail_end)
        } else {
            let tail_end = prefix_end + count * 2;
            if data.len() < tail_end { return None; }
            let ti: Vec<u16> = (0..count).map(|i| {
                let off = prefix_end + i * 2;
                u16::from_le_bytes(data[off..off+2].try_into().unwrap_or([0, 0]))
            }).collect();
            (ti, tail_end)
        };

        let sign_bytes = (count + 7) / 8;
        if data.len() < sign_start + sign_bytes { return None; }
        let sign_bits = data[sign_start..sign_start + sign_bytes].to_vec();
        Some(Self { prefix_idx, tail_idx, tail_width, sign_bits, count })
    }

    pub fn bytes(&self) -> usize {
        4 + 1 + self.prefix_idx.len()
        + self.tail_idx.len() * if self.tail_width == 0 { 1 } else { 2 }
        + self.sign_bits.len()
    }

    /// Pack sandbag into GPU-friendly u32 array.
    /// Each u32: {prefix_idx:8, tail_idx:16, sign:1, pad:7}
    /// Layout: bits 0-7 = prefix_idx, bits 8-23 = tail_idx, bit 24 = sign
    /// This is the format consumed by the WGSL decompression kernel.
    pub fn pack_for_gpu(&self) -> Vec<u32> {
        let mut packed = Vec::with_capacity(self.count);
        for i in 0..self.count {
            let p = self.prefix_idx.get(i).copied().unwrap_or(0) as u32;
            let t = self.tail_idx.get(i).copied().unwrap_or(0) as u32;
            let sign = ((self.sign_bits.get(i / 8).copied().unwrap_or(0) >> (i % 8)) & 1) as u32;
            packed.push(p | (t << 8) | (sign << 24));
        }
        packed
    }
}

/// Compressed tensor — the core dictionary. This is ALL that gets loaded.
///
/// Bidirectional layout (front to back):
///   [prefix_0][prefix_1]...[prefix_N]   ← unique prefix values (front)
///   [tail_0][tail_1]...[tail_M]          ← unique tail values
///   [GapFlag]
///   [tail_count_M]...[tail_count_0]      ← per-tail counts (reversed, back)
///   [TailFlag]
///   [prefix_count_N]...[prefix_count_0]  ← per-prefix counts (reversed, very back)
///   [CountFlag]
///
/// Pairing: prefix[i] at front position [i] ↔ prefix_count[i] at back position [-(i+1)]
/// Same for tails. Oppositely aligned by index.
#[derive(Clone, Debug)]
pub struct DedupCountTensor {
    pub prefixes: Vec<u16>,      // integer prefixes (e.g., 25 for 0.25 with 2-digit)
    pub prefix_counts: Vec<u32>,
    pub unique_tails: Vec<UniqueTail>,
    pub count: usize,
    pub prefix_digits: usize,
    pub tail_digits: usize,
    pub avg_precision_lost: f32,
}

impl DedupCountTensor {
    const TOTAL_DIGITS: usize = 7;

    pub fn compress(weights: &[f32], prefix_digits: usize, truncate_rounds: usize) -> (Self, Sandbag) {
        let initial_tail_digits = Self::TOTAL_DIGITS - prefix_digits;
        let prefix_scale = 10f32.powi(prefix_digits as i32);
        let tail_scale = 10f32.powi(Self::TOTAL_DIGITS as i32);
        let n = weights.len();

        // Step 1: Group by prefix — integer prefix eliminates f32 precision bugs.
        // prefix_int = floor(abs_w * scale) as u16 (e.g., 25 for 0.25)
        let mut prefix_map: AHashMap<u16, u8> = AHashMap::with_capacity(256);
        let mut prefixes: Vec<u16> = Vec::new();
        let mut prefix_idx = vec![0u8; n];
        let mut sign_bits = vec![0u8; (n + 7) / 8];
        let mut group_tails: Vec<Vec<u32>> = Vec::new();

        for (i, &w) in weights.iter().enumerate() {
            let sign = w < 0.0;
            let abs_w = w.abs();
            let prefix_int = (abs_w * prefix_scale).floor() as u16;
            let prefix_val = prefix_int as f32 / prefix_scale;
            let tail_val = abs_w - prefix_val;
            let tail_int = (tail_val * tail_scale).round() as u32;

            let group_idx = match prefix_map.get(&prefix_int) {
                Some(&idx) => idx,
                None if prefix_map.len() < 256 => {
                    let idx = prefix_map.len() as u8;
                    prefix_map.insert(prefix_int, idx);
                    prefixes.push(prefix_int);
                    group_tails.push(Vec::new());
                    idx
                }
                _ => 0,
            };

            prefix_idx[i] = group_idx;
            if sign { sign_bits[i / 8] |= 1 << (i % 8); }
            group_tails[group_idx as usize].push(tail_int);
        }

        // Step 2: Truncation with averaging (operates on group_tails — small)
        let mut global_loss_sum = 0.0f32;
        let mut global_loss_count = 0usize;
        let mut current_tail_digits = initial_tail_digits;
        let mut round_ups: Vec<Vec<bool>> = vec![Vec::with_capacity(truncate_rounds); prefixes.len()];

        for _round in 0..truncate_rounds {
            let current_divisor = 10f32.powi((prefix_digits + current_tail_digits) as i32);
            let next_divisor = 10f32.powi((prefix_digits + current_tail_digits - 1) as i32);

            for (gidx, gt) in group_tails.iter_mut().enumerate() {
                let last_digits: Vec<u32> = gt.iter().map(|t| t % 10).collect();
                let avg = last_digits.iter().sum::<u32>() as f32
                    / last_digits.len().max(1) as f32;
                let round_up = avg > 5.0;
                round_ups[gidx].push(round_up);

                for tail in gt.iter_mut() {
                    let old_val = *tail as f32 / current_divisor;
                    *tail = if round_up { *tail / 10 + 1 } else { *tail / 10 };
                    let new_val = *tail as f32 / next_divisor;
                    global_loss_sum += (old_val - new_val).abs();
                    global_loss_count += 1;
                }
            }
            current_tail_digits -= 1;
        }

        let global_avg_lost = global_loss_sum / global_loss_count.max(1) as f32;

        // Step 3: Find unique tails + count — O(n) with HashMap (was O(n²) with linear scan)
        let mut tail_counts: AHashMap<u16, u32> = AHashMap::new();
        for gt in &group_tails {
            for &tail in gt {
                let tv = tail as u16;
                *tail_counts.entry(tv).or_insert(0) += 1;
            }
        }

        // Sort for deterministic output
        let mut unique_tail_values: Vec<u16> = tail_counts.keys().copied().collect();
        unique_tail_values.sort_unstable();

        // Build tail → index map — O(unique_count)
        let tail_idx_map: AHashMap<u16, u16> = unique_tail_values.iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u16))
            .collect();

        let unique_tails: Vec<UniqueTail> = unique_tail_values.iter().map(|&v| {
            UniqueTail { value: v, repeat_count: tail_counts[&v] }
        }).collect();

        // Step 4: Build tail_idx — reconstruct from integer prefix (no f32 prefix storage)
        let mut tail_idx = vec![0u16; n];
        for (i, &w) in weights.iter().enumerate() {
            let abs_w = w.abs();
            let prefix_int = (abs_w * prefix_scale).floor() as u16;
            let prefix_val = prefix_int as f32 / prefix_scale;
            let tail_val = abs_w - prefix_val;
            let mut tail_int = (tail_val * tail_scale).round() as u32;
            let gidx = prefix_idx[i] as usize;
            for round in 0..truncate_rounds {
                let ru = round_ups[gidx].get(round).copied().unwrap_or(false);
                tail_int = if ru { tail_int / 10 + 1 } else { tail_int / 10 };
            }
            let tv = tail_int as u16;
            tail_idx[i] = tail_idx_map.get(&tv).copied().unwrap_or(0);
        }

        let prefix_counts: Vec<u32> = (0..prefixes.len())
            .map(|gidx| group_tails[gidx].len() as u32)
            .collect();

        let tensor = Self {
            prefixes, prefix_counts, unique_tails,
            count: n, prefix_digits,
            tail_digits: current_tail_digits,
            avg_precision_lost: global_avg_lost,
        };

        let tail_width: u8 = if truncate_rounds >= 3 { 0 } else { 1 };
        let sandbag = Sandbag { prefix_idx, tail_idx, tail_width, sign_bits, count: n };
        (tensor, sandbag)
    }

    /// GPU-accelerated compress. Takes pre-computed per-element arrays from
    /// the GPU shader (prefix_bits, tails, signs). CPU does grouping,
    /// truncation, and dedup using HashMap — O(n) total.
    pub fn compress_from_gpu(
        weights: &[f32],
        prefix_bits: &[u32],
        tails: &[u32],
        signs: &[u32],
        prefix_digits: usize,
        truncate_rounds: usize,
    ) -> (Self, Sandbag) {
        let initial_tail_digits = Self::TOTAL_DIGITS - prefix_digits;
        let prefix_scale = 10f32.powi(prefix_digits as i32);
        let n = weights.len();

        // Step 1: Group by integer prefix
        let mut prefix_map: AHashMap<u16, u8> = AHashMap::with_capacity(256);
        let mut prefixes: Vec<u16> = Vec::new();
        let mut prefix_idx = vec![0u8; n];
        let mut sign_bits = vec![0u8; (n + 7) / 8];
        let mut group_tails: Vec<Vec<u32>> = Vec::new();

        for i in 0..n {
            let pv = f32::from_bits(prefix_bits[i]);
            let prefix_int = (pv * prefix_scale).floor() as u16;
            let group_idx = match prefix_map.get(&prefix_int) {
                Some(&idx) => idx,
                None if prefix_map.len() < 256 => {
                    let idx = prefix_map.len() as u8;
                    prefix_map.insert(prefix_int, idx);
                    prefixes.push(prefix_int);
                    group_tails.push(Vec::new());
                    idx
                }
                _ => 0,
            };

            prefix_idx[i] = group_idx;
            if signs[i] != 0 { sign_bits[i / 8] |= 1 << (i % 8); }
            group_tails[group_idx as usize].push(tails[i]);
        }

        // Step 2: Truncation with averaging (CPU reduction — operates on small group_tails)
        let mut global_loss_sum = 0.0f32;
        let mut global_loss_count = 0usize;
        let mut current_tail_digits = initial_tail_digits;
        let mut round_ups: Vec<Vec<bool>> = vec![Vec::with_capacity(truncate_rounds); prefixes.len()];

        for _round in 0..truncate_rounds {
            let current_divisor = 10f32.powi((prefix_digits + current_tail_digits) as i32);
            let next_divisor = 10f32.powi((prefix_digits + current_tail_digits - 1) as i32);

            for (gidx, gt) in group_tails.iter_mut().enumerate() {
                let last_digits: Vec<u32> = gt.iter().map(|t| t % 10).collect();
                let avg = last_digits.iter().sum::<u32>() as f32
                    / last_digits.len().max(1) as f32;
                let round_up = avg > 5.0;
                round_ups[gidx].push(round_up);

                for tail in gt.iter_mut() {
                    let old_val = *tail as f32 / current_divisor;
                    *tail = if round_up { *tail / 10 + 1 } else { *tail / 10 };
                    let new_val = *tail as f32 / next_divisor;
                    global_loss_sum += (old_val - new_val).abs();
                    global_loss_count += 1;
                }
            }
            current_tail_digits -= 1;
        }

        let global_avg_lost = global_loss_sum / global_loss_count.max(1) as f32;

        // Step 3: Find unique tails + count — O(n) with HashMap (was O(n²) with linear scan)
        let mut tail_counts: AHashMap<u16, u32> = AHashMap::new();
        for gt in &group_tails {
            for &tail in gt {
                let tv = tail as u16;
                *tail_counts.entry(tv).or_insert(0) += 1;
            }
        }

        let mut unique_tail_values: Vec<u16> = tail_counts.keys().copied().collect();
        unique_tail_values.sort_unstable();

        let tail_idx_map: AHashMap<u16, u16> = unique_tail_values.iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u16))
            .collect();

        let unique_tails: Vec<UniqueTail> = unique_tail_values.iter().map(|&v| {
            UniqueTail { value: v, repeat_count: tail_counts[&v] }
        }).collect();

        // Step 4: Build tail_idx — O(n) with HashMap lookup (was O(n × unique_count))
        let mut tail_idx = vec![0u16; n];
        for i in 0..n {
            let mut tail_int = tails[i];
            let gidx = prefix_idx[i] as usize;
            for round in 0..truncate_rounds {
                let ru = round_ups[gidx].get(round).copied().unwrap_or(false);
                tail_int = if ru { tail_int / 10 + 1 } else { tail_int / 10 };
            }
            let tv = tail_int as u16;
            tail_idx[i] = tail_idx_map.get(&tv).copied().unwrap_or(0);
        }

        let prefix_counts: Vec<u32> = (0..prefixes.len())
            .map(|gidx| group_tails[gidx].len() as u32)
            .collect();

        let tensor = Self {
            prefixes, prefix_counts, unique_tails,
            count: n, prefix_digits,
            tail_digits: current_tail_digits,
            avg_precision_lost: global_avg_lost,
        };

        let tail_width: u8 = if truncate_rounds >= 3 { 0 } else { 1 };
        let sandbag = Sandbag { prefix_idx, tail_idx, tail_width, sign_bits, count: n };
        (tensor, sandbag)
    }

    /// Decompress using sandbag map (stays on disk, loaded by worker).
    /// Precision recovery: add avg_precision_lost to each tail.
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

    /// Zero-copy decompression using the global prefix→tail table.
    /// Workers read indices from sandbag, dereference into GlobalTable —
    /// no per-chunk prefix/tail value loading needed.
    pub fn decompress_all_global(&self, sandbag: &Sandbag, global: &GlobalTable) -> Vec<f32> {
        // Tail values come from the global table, so the divisor must use the GLOBAL table's
        // prefix_digits (not the chunk's) combined with the chunk's tail_digits.
        let tail_divisor = 10f32.powi((global.prefix_digits + self.tail_digits) as i32);
        let gt_scale = 10f32.powi(global.prefix_digits as i32);
        let scale_diff = global.prefix_digits as i32 - self.prefix_digits as i32;

        // Build per-chunk → global prefix index map using integer prefixes
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

        // Build per-chunk tail index → global tail index map
        let mut tail_lookup: AHashMap<(u8, u16), usize> = AHashMap::new();
        for (cp_idx, &pv) in self.prefixes.iter().enumerate() {
            let gp_idx = *prefix_lookup.get(&pv).unwrap_or(&0);
            let global_tails = &global.tails_for_prefix[gp_idx];
            let mut val_to_gi: AHashMap<u32, usize> = AHashMap::with_capacity(global_tails.len());
            for (gi, &t) in global_tails.iter().enumerate() {
                val_to_gi.insert(t, gi);
            }
            for (ct_idx, ut) in self.unique_tails.iter().enumerate() {
                if let Some(&gi) = val_to_gi.get(&(ut.value as u32)) {
                    tail_lookup.insert((cp_idx as u8, ct_idx as u16), gi);
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

    /// Core size — bidirectional layout. Only unique values + counts.
    /// This is what gets loaded. Sandbag stays on disk.
    pub fn compressed_bytes(&self) -> usize {
        // Header
        let header = 4 + 4 + 4 + 4 + 4; // count, prefix_digits, tail_digits, avg_pl, group_count
        // Front: prefix values + tail values
        let front = self.prefixes.len() * 2    // u16 per prefix (integer, one instance)
            + self.unique_tails.len() * 2;     // u16 per tail value (one instance)
        // Flags
        let flags = 3; // GapFlag, TailFlag, CountFlag
        // Back: tail counts (reversed) + prefix counts (reversed)
        let back = self.unique_tails.len() * 4  // u32 repeat_count per tail
            + self.prefix_counts.len() * 4;     // u32 per prefix count
        header + front + flags + back
    }

    pub fn original_bytes(&self) -> usize { self.count * 4 }

    pub fn ratio(&self) -> f32 {
        let comp = self.compressed_bytes() as f32;
        if comp == 0.0 { 1.0 } else { self.original_bytes() as f32 / comp }
    }

    pub fn unique_tail_count(&self) -> usize { self.unique_tails.len() }
    pub fn unique_prefix_count(&self) -> usize { self.prefixes.len() }

    pub fn shared_tail_weights(&self) -> usize {
        self.unique_tails.iter()
            .filter(|t| t.repeat_count > 1)
            .map(|t| t.repeat_count as usize)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_compression() {
        let mut weights = Vec::new();
        for i in 0..10_000 {
            let base = match i % 4 {
                0 => 0.0150, 1 => 0.0230, 2 => -0.0180, _ => 0.0420,
            };
            let noise = ((i * 7) as f32 * 0.00017).fract() * 0.0009;
            weights.push(base + noise + if i % 2 == 0 { 0.0 } else { 0.0001 });
        }

        let (packed, sandbag) = DedupCountTensor::compress(&weights, 2, 2);
        let decompressed = packed.decompress_all(&sandbag);
        let max_err = weights.iter().zip(decompressed.iter())
            .map(|(o, d)| (o - d).abs()).fold(0.0f32, f32::max);

        println!("\nDedup compression (10K weights, 2 rounds):");
        println!("  Unique prefixes: {} (one instance each)", packed.unique_prefix_count());
        println!("  Unique tails: {} (one instance each)", packed.unique_tail_count());
        println!("  Core: {} B, Sandbag (on disk): {} B, Original: {} B",
            packed.compressed_bytes(), sandbag.bytes(), packed.original_bytes());
        println!("  Core ratio: {:.0}x  (sandbag not counted — stays on disk)",
            packed.ratio());
        println!("  Max error: {:.6}", max_err);
        assert!(max_err < 0.1);
    }

    #[test]
    fn test_large_dedup() {
        let mut weights = Vec::with_capacity(100_000);
        for i in 0..100_000 {
            let base = match i % 5 {
                0 => 0.0150, 1 => 0.0230, 2 => -0.0180, 3 => 0.0420, _ => -0.0310,
            };
            let noise = ((i * 13) as f32 * 0.00007).fract() * 0.0009;
            weights.push(base + noise);
        }

        let (packed, sandbag) = DedupCountTensor::compress(&weights, 2, 2);
        let decompressed = packed.decompress_all(&sandbag);
        let max_err = weights.iter().zip(decompressed.iter())
            .map(|(o, d)| (o - d).abs()).fold(0.0f32, f32::max);

        println!("\nLarge dedup (100K weights, 2 rounds):");
        println!("  Unique prefixes: {}, Unique tails: {}", packed.unique_prefix_count(), packed.unique_tail_count());
        println!("  Core: {} B, Sandbag (on disk): {} B, Original: {} B",
            packed.compressed_bytes(), sandbag.bytes(), packed.original_bytes());
        println!("  Core ratio: {:.0}x  (sandbag stays on disk, not loaded)",
            packed.ratio());
        println!("  Max error: {:.6}", max_err);
        assert!(max_err < 0.1);
    }

    #[test]
    fn test_aggressive_truncation() {
        let mut weights = Vec::with_capacity(100_000);
        for i in 0..100_000 {
            let base = match i % 5 {
                0 => 0.0150, 1 => 0.0230, 2 => -0.0180, 3 => 0.0420, _ => -0.0310,
            };
            let noise = ((i * 13) as f32 * 0.00007).fract() * 0.0009;
            weights.push(base + noise);
        }

        // 3 rounds: tails drop to 2 digits (max 99) → massive dedup
        let (packed, sandbag) = DedupCountTensor::compress(&weights, 2, 3);
        let decompressed = packed.decompress_all(&sandbag);
        let max_err = weights.iter().zip(decompressed.iter())
            .map(|(o, d)| (o - d).abs()).fold(0.0f32, f32::max);

        println!("\nAggressive (100K weights, 3 rounds):");
        println!("  Unique prefixes: {}, Unique tails: {}", packed.unique_prefix_count(), packed.unique_tail_count());
        println!("  Core: {} B, Sandbag (disk): {} B, Original: {} B",
            packed.compressed_bytes(), sandbag.bytes(), packed.original_bytes());
        println!("  Core ratio: {:.0}x  (sandbag not loaded)",
            packed.ratio());
        println!("  Max error: {:.6}", max_err);
        assert!(max_err < 0.1);
    }

    #[test]
    fn test_sandbag_roundtrip() {
        let mut weights = Vec::new();
        for i in 0..1_000 {
            let base = match i % 3 { 0 => 0.0150, 1 => -0.0230, _ => 0.0420 };
            let noise = ((i * 7) as f32 * 0.00017).fract() * 0.0009;
            weights.push(base + noise);
        }
        let (packed, sandbag) = DedupCountTensor::compress(&weights, 2, 2);
        let bytes = sandbag.to_bytes();
        let restored = Sandbag::from_bytes(&bytes).expect("deserialization failed");
        let d1 = packed.decompress_all(&sandbag);
        let d2 = packed.decompress_all(&restored);
        let max_diff = d1.iter().zip(d2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff == 0.0, "sandbag round-trip mismatch: {}", max_diff);
    }

    #[test]
    fn test_truncation_averaging() {
        let weights: Vec<f32> = vec![0.157, 0.257, 0.357, 0.457];
        let (packed, sandbag) = DedupCountTensor::compress(&weights, 1, 2);
        let decompressed = packed.decompress_all(&sandbag);
        let max_err = weights.iter().zip(decompressed.iter())
            .map(|(o, d)| (o - d).abs()).fold(0.0f32, f32::max);
        println!("\nTruncation: weights={:?} decompressed={:?} max_err={:.6}", weights, decompressed, max_err);
        assert!(max_err < 0.1);
    }
}

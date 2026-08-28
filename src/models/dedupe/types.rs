#[derive(Clone, Copy, Debug)]
pub enum DataFlag {
	GapFlag = 0xFD,
	TailFlag = 0xFE,
	CountFlag = 0xFF,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UniqueTail {
	pub value: u32,
	pub repeat_count: u32,
}

/// Compressed representation of a weight block.
///
/// Values are split into prefix_int + tail_int (matching GPU/AVX shader math):
///   prefix_int = floor(abs_w * 10^prefix_digits)
///   tail_int   = round((abs_w - prefix_int/10^prefix_digits) * 10^7)
///
/// Reconstruction:
///   abs_w = (prefix_int as f32) / 10^prefix_digits + (tail_int as f32) / 10^7
///   w = if sign_bit_set { -abs_w } else { abs_w }
///
/// Outliers stored at full precision.
#[derive(Clone, Debug)]
pub struct Sandbag {
	/// Per-block scale factor (reserved for future use; kept for compatibility)
	pub scale: f32,
	/// Outlier positions and original f32 values
	pub outliers: Vec<(usize, f32)>,
	/// Total element count
	pub count: usize,
	/// Number of prefix digits used during compression
	pub prefix_digits: usize,
	/// Deduped unique prefix_int values (u8 — max 256 unique prefixes)
	pub unique_prefixes: Vec<u8>,
	/// Deduped unique tail_int values (u32 — full range 0..9,999,999)
	pub unique_tails: Vec<u32>,
	/// Manifest: per-element (prefix_idx, tail_idx) for reconstruction.
	/// prefix_idx and tail_idx index into unique_prefixes and unique_tails.
	pub manifest: Vec<(u16, u16)>,
	/// Sign bitvector — bit i is 1 if element i is negative, 0 if positive.
	pub signs: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct GlobalTable {
	pub prefix_digits: usize,
	pub prefixes: Vec<f32>,
	pub tails_for_prefix: Vec<Vec<u32>>,
}

#[derive(Clone, Debug)]
pub struct ChunkRemap {
	pub global_tail_indices: Vec<u16>,
}

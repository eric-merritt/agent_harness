#[derive(Clone, Debug)]
pub struct Sandbag {
	/// Block size buffer inclusion
	pub size: BlockSizeBuffer,
	/// Outlier positions and original f32 values
	pub meta_tensors: Vec<(usize, f32)>,
	/// Total element count
	pub count: usize,
	/// Number of prefix digits used during compression
	pub prefix_digits: usize,
	/// Deduped unique prefix_int values (u8 — max 256 unique prefixes)
	pub unique_prefixes: Vec<u8>,
	/// Deduped unique tail_int values (u32 — full range 0..9,999,999)
	pub unique_tails: Vec<[u16; 4]>,
	pub index_signs: Vec<(u16,u16)>,
}

/// Packed block dimensions: high 16 bits = width, low 16 bits = height.
#[repr(transparent)]
#[derive(Clone, Copy, Default, Debug)]
pub struct BlockSizeBuffer(pub u32);

impl BlockSizeBuffer {
	pub fn new(width: u16, height: u16) -> Self {
		Self((u32::from(width) << 16) | u32::from(height))
	}
	pub fn width(&self) -> u16 {
		(self.0 >> 16) as u16
	}
	pub fn height(&self) -> u16 {
		(self.0 & 0xFFFF) as u16
	}
	/// Total elements in this block (width × height).
	pub fn block_size(&self) -> usize {
		(self.width() as usize) * (self.height() as usize)
	}
}


	/// Single bucket entry written by the shader.
/// `prefix_idx` is the u8 deduplicated prefix index.
/// `tails` is a fixed 4-entry array of u16 tail indices.
#[repr(C, align(4))]
pub struct BucketEntry {
	pub prefix_idx: u8,
	/// 4 tail indices, each a u16.
	pub tails: [u16; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct UniqueTail {
	pub value: u32,
	pub repeat_count: u32,
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

impl Sandbag {
	pub fn save(&self, ){

	}

	
}
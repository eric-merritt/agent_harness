#[derive(Clone, Copy, Debug)]
pub enum DataFlag {
    GapFlag = 0xFD,
    TailFlag = 0xFE,
    CountFlag = 0xFF,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UniqueTail {
    pub value: u16,
    pub repeat_count: u32,
}

#[derive(Clone, Debug)]
pub struct Sandbag {
    pub prefix_idx: Vec<u16>, 
    pub tail_idx: Vec<u16>,
    pub tail_width: u8,
    pub sign_bits: Vec<u8>,
    pub count: usize,
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
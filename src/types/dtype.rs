// Data type enumeration — f32, f16, bf16, quantized types.
// Block-based quantization (Q4_0, Q8_0) stores a scale per group of N elements.

use half::{bf16, f16};
use std::mem;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
    F16,
    BF16,
    I8,
    U8,
    Q8_0,
    Q4_0,
}

impl DataType {
    /// Number of elements per quantization block (1 for unquantized types).
    pub fn block_size(&self) -> usize {
        match self {
            DataType::F32 | DataType::BF16 | DataType::F16
            | DataType::I8 | DataType::U8 => 1,
            DataType::Q8_0 | DataType::Q4_0 => 32,
        }
    }

    /// Total bytes for one quantization block.
    pub fn block_size_in_bytes(&self) -> usize {
        match self {
            DataType::F32 => mem::size_of::<f32>(),
            DataType::F16 => mem::size_of::<f16>(),
            DataType::BF16 => mem::size_of::<bf16>(),
            DataType::I8 => mem::size_of::<i8>(),
            DataType::U8 => mem::size_of::<u8>(),
            DataType::Q4_0 => mem::size_of::<BlockQ4_0>(),
            DataType::Q8_0 => mem::size_of::<BlockQ8_0>(),
        }
    }

    /// Alignment requirement in bytes.
    pub fn align(&self) -> usize {
        match self {
            DataType::F32 => mem::align_of::<f32>(),
            DataType::F16 => mem::align_of::<f16>(),
            DataType::BF16 => mem::align_of::<bf16>(),
            DataType::I8 => mem::align_of::<i8>(),
            DataType::U8 => mem::align_of::<u8>(),
            DataType::Q4_0 => mem::align_of::<BlockQ4_0>(),
            DataType::Q8_0 => mem::align_of::<BlockQ8_0>(),
        }
    }

    /// Bytes per single element (for unquantized types).
    pub fn byte_size(&self) -> usize {
        self.block_size_in_bytes() / self.block_size()
    }

    /// Total bytes for `num_elements` of this dtype.
    pub fn total_bytes(&self, num_elements: usize) -> usize {
        let blocks = (num_elements + self.block_size() - 1) / self.block_size();
        blocks * self.block_size_in_bytes()
    }
}

// ── Quantization block layouts ───────────────────────────────────────────────

/// Q4_0 block: 1 f16 scale + 16 packed bytes (32 × 4-bit indices).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub delta: f16,
    pub quantized_symbols: [u8; 16],
}

/// Q8_0 block: 1 f16 scale + 32 int8 values.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub delta: f16,
    pub quantized_symbols: [i8; 32],
}

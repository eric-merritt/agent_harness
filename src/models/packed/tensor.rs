use std::collections::HashMap;

/// A compressed weight tensor using coordinate + sign-inverse compression
/// with alignment-packing.
#[derive(Clone, Debug)]
pub struct PackedTensor {
    /// Cluster prefix values (f32, one per cluster).
    pub prefixes: Vec<f32>,
    /// Tail tables — one vec of f32 per cluster.
    pub tail_tables: Vec<Vec<f32>>,
    /// Packed data: for each weight, [cluster_byte | tail_byte]
    /// where cluster_byte has the inverse bit in bit 7.
    pub packed: Vec<u8>,
    /// Weights that couldn't be compressed (too many clusters or tails).
    /// Stored as full f32 values, indexed by their position in the weight array.
    pub uncompressed: HashMap<usize, f32>,
    /// Original element count.
    pub count: usize,
    /// Alignment in bytes (typically 8 for safetensors, 32 for GGUF).
    pub alignment: usize,
}
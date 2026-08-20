use crate::inference::config::ModelConfig;

/// KV cache for GQA attention layers.
///
/// Layout: K and V are each `[n_head_kv, head_dim, max_seq_len]` (flat, row-major).
/// For head `h`, dimension `d`, position `p`:
///   index = h * head_dim * max_seq_len + d * max_seq_len + p
pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl KvCache {
    pub fn max_seq_len(&self) {
        &self.max_seq_len;
    }
    /// Allocate a KV cache sized for the given model config.
    pub fn new(config: &ModelConfig) -> Self {
        let n_head_kv = config.n_head_kv;
        let head_dim = config.n_embd_head;
        let max_seq_len: usize = 4096;
        let size = n_head_kv * head_dim * max_seq_len;
        
        // Zero-initialized to guarantee safety and prevent uninitialized heap page reads (UB)
        Self {
            k: vec![0.0f32; size],
            v: vec![0.0f32; size],
            n_head_kv,
            head_dim,
            max_seq_len,
        }
    }

    /// Create an empty (zero-capacity) cache for non-attention (SSM) layers.
    pub fn empty() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            n_head_kv: 0,
            head_dim: 0,
            max_seq_len: 0,
        }
    }

    /// Write K for KV-head `h` at sequence position `pos`.
    pub fn write_k(&mut self, h: usize, pos: usize, k_head: &[f32]) {
        let base = h * self.head_dim * self.max_seq_len;
        for d in 0..self.head_dim {
            self.k[base + d * self.max_seq_len + pos] = k_head[d];
        }
    }

    /// Write V for KV-head `h` at sequence position `pos`.
    pub fn write_v(&mut self, h: usize, pos: usize, v_head: &[f32]) {
        let base = h * self.head_dim * self.max_seq_len;
        for d in 0..self.head_dim {
            self.v[base + d * self.max_seq_len + pos] = v_head[d];
        }
    }
}
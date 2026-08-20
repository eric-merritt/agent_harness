pub mod gguf;
pub mod lora;
pub mod pt;
pub mod safetensors;
pub mod quantization;
pub mod packed_weights;
pub mod count_indexed;
pub mod dedup_count;
pub mod convert;
pub mod server;

use crate::augment::augment::{TensorDescriptor, TensorDtype, ModelTensorMap};

/// Trait implemented by every model file format adapter.
/// Each adapter knows how to read its format and expose a unified tensor view.
pub trait ModelAdapter {
    /// The model name or identifier.
    fn model_name(&self) -> &str;

    /// Total number of tensors in the model.
    fn tensor_count(&self) -> usize;

    /// Total byte size of all tensor data.
    fn total_bytes(&self) -> u64;

    /// Get tensor descriptor at a given index.
    fn tensor_at(&self, index: usize) -> Option<TensorDescriptor>;

    /// Find a tensor by name.
    fn find_tensor(&self, name: &str) -> Option<TensorDescriptor>;

    /// Byte alignment required by this format (8 for safetensors, 32 for GGUF, etc.)
    fn alignment(&self) -> usize;

    /// Raw tensor data slice for a given tensor name.
    fn raw_data(&self, tensor_name: &str) -> Option<&[u8]>;

    /// Convert the entire adapter into a unified ModelTensorMap for the AugmentBus.
    fn to_tensor_map(&self) -> ModelTensorMap {
        let tensors: Vec<TensorDescriptor> = (0..self.tensor_count())
            .filter_map(|i| self.tensor_at(i))
            .collect();
        let total = tensors.iter().map(|t| t.byte_size).sum();
        ModelTensorMap {
            model_name: self.model_name().to_string(),
            tensors,
            total_bytes: total,
        }
    }
}

/// Convert a dtype string (from safetensors/JSON) to the unified TensorDtype.
pub fn parse_dtype(s: &str) -> TensorDtype {
    match s.to_uppercase().as_str() {
        "F32" | "FLOAT32" => TensorDtype::F32,
        "F16" | "FLOAT16" => TensorDtype::F16,
        "BF16" | "BFLOAT16" => TensorDtype::BF16,
        "I8" | "INT8" => TensorDtype::I8,
        "I16" | "INT16" => TensorDtype::I16,
        "I32" | "INT32" => TensorDtype::I32,
        "I64" | "INT64" => TensorDtype::I64,
        "U8" | "UINT8" => TensorDtype::U8,
        "BOOL" => TensorDtype::Bool,
        _ => TensorDtype::F32, // safe default
    }
}
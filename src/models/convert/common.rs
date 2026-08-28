// Shared conversion types, parameters, and weight-compression entry points.
//
// Used by the GGUF, safetensors, and quantization conversion pipelines.

pub use crate::models::avx512_kernel::avx512_preprocess_conversion_chunk;
use crate::models::dedupe::tensor::DedupCountTensor;
use crate::models::quantization::QuantizationLevels;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorStats {
	pub name: String,
	pub shape: Vec<usize>,
	pub gguf_dtype: u32,
	pub element_count: usize,
	pub original_bytes: usize,
	pub core_bytes: usize,
	pub sandbag_bytes: usize,
	pub core_ratio: f32,
	pub prefix_count: usize,
	pub unique_tail_count: usize,
	pub shared_weights: usize,
	pub mean_precision_lost: f32,
	pub weight_offset: u64,
	pub sandbag_offset: u64,
	pub full_precision: bool,
	#[serde(default)]
	pub quant_offset: u64,
	#[serde(default)]
	pub quant_bytes: usize,
	#[serde(default)]
	pub is_4bit: bool,
	#[serde(default = "default_group_size")]
	pub group_size: usize,
}

fn default_group_size() -> usize {
	32
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConversionStats {
	pub model_name: String,
	pub tensor_count: usize,
	pub total_original_bytes: u64,
	pub total_core_bytes: u64,
	pub total_sandbag_bytes: u64,
	pub overall_core_ratio: f32,
	pub tensors: Vec<TensorStats>,
}

pub const HIGH_PRECISION_TENSORS: &[&str] = &["output.weight", "lm_head.weight"];

pub const HIGH_PRECISION_EXTRA_DIGITS: usize = 0;
pub const HIGH_PRECISION_TRUNCATE_ROUNDS: usize = 0;
pub const CHUNK_SIZE: usize = 1_000_000;
pub const GPU_MIN_ELEMENTS: usize = 100_000;

/// Resolve quantization level and params from a tensor name.
pub fn resolve_quantization(name: &str) -> QuantizationLevels {
	QuantizationLevels::from_name(name)
}

pub fn resolve_params(name: &str, prefix_digits: usize, truncate_rounds: usize) -> (usize, usize) {
	let level = resolve_quantization(name);
	match level {
		QuantizationLevels::FullPrecision => (0, 0),
		QuantizationLevels::HalfPrecision => (prefix_digits + 2, 0),
		QuantizationLevels::ToNeg4 => (prefix_digits, truncate_rounds),
		QuantizationLevels::ToNeg8 => (prefix_digits, truncate_rounds),
	}
}

pub struct CompressOutput {
	pub core: Vec<u8>,
	pub sandbag: Vec<u8>,
	pub prefix_count: usize,
	pub unique_tail_count: usize,
	pub shared_weights: usize,
	pub mean_precision_lost: f32,
	pub full_precision: bool,
}

pub struct CompressJob {
	pub global_idx: usize,
	pub name: String,
	pub shape: Vec<usize>,
	pub element_count: usize,
	pub weights: Vec<f32>,
}

#[derive(Clone)]
pub struct CompressResult {
	pub global_idx: usize,
	pub stats: TensorStats,
	pub core: Vec<u8>,
	pub sandbag: Vec<u8>,
}

impl ConversionStats {
	pub fn to_binary_cache(&self, dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
		let bin_path = dir.join("manifest.bin");
		if let Ok(data) = bincode::serialize(self) {
			let _ = std::fs::write(bin_path, data);
		}
		Ok(())
	}
}

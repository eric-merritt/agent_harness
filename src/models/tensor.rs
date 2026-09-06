use std::fmt;

use crate::models::format::GgmlType;

/// Every tensor type the harness knows about.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorType {
	// Attention
	QKVWeight,
	QKVBias,
	AttnOutWeight,
	AttnOutBias,
	RopeFreqs,
	// Feed-forward
	GateWeight,
	UpWeight,
	DownWeight,
	GateBias,
	UpBias,
	DownBias,
	// Normalization
	RmsNormWeight,
	RmsNormBias,
	LayerNormWeight,
	LayerNormBias,
	// Activations
	ActivationSine,
	ActivationCosine,
	// Output
	OutputWeight,
	OutputBias,
	// Embedding
	EmbeddingWeight,
	// Generic / catch-all
	Weight { layer: usize, kind: String },
	Bias { layer: usize },
	Unknown,
}

impl fmt::Display for TensorType {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			TensorType::QKVWeight => write!(f, "qkv_weight"),
			TensorType::QKVBias => write!(f, "qkv_bias"),
			TensorType::AttnOutWeight => write!(f, "attn_out_weight"),
			TensorType::AttnOutBias => write!(f, "attn_out_bias"),
			TensorType::RopeFreqs => write!(f, "rope_freqs"),
			TensorType::GateWeight => write!(f, "gate_weight"),
			TensorType::UpWeight => write!(f, "up_weight"),
			TensorType::DownWeight => write!(f, "down_weight"),
			TensorType::GateBias => write!(f, "gate_bias"),
			TensorType::UpBias => write!(f, "up_bias"),
			TensorType::DownBias => write!(f, "down_bias"),
			TensorType::RmsNormWeight => write!(f, "rms_norm_weight"),
			TensorType::RmsNormBias => write!(f, "rms_norm_bias"),
			TensorType::LayerNormWeight => write!(f, "layer_norm_weight"),
			TensorType::LayerNormBias => write!(f, "layer_norm_bias"),
			TensorType::ActivationSine => write!(f, "activation_sine"),
			TensorType::ActivationCosine => write!(f, "activation_cosine"),
			TensorType::OutputWeight => write!(f, "output_weight"),
			TensorType::OutputBias => write!(f, "output_bias"),
			TensorType::EmbeddingWeight => write!(f, "embedding_weight"),
			TensorType::Weight { layer, kind } => write!(f, "layer_{}_{}", layer, kind),
			TensorType::Bias { layer } => write!(f, "layer_{}_bias", layer),
			TensorType::Unknown => write!(f, "unknown"),
		}
	}
}

/// Dimensionality of a tensor (up to 8D covers everything practical).
pub type Dim = Vec<usize>;

/// Re-export GgmlType as the local dtype for tensors.
pub type QuantScheme = GgmlType;

/// Format-agnostic tensor descriptor.
#[derive(Debug, Clone)]
pub struct Tensor {
	/// Original name from the source file (e.g. "model.layers.0.self_attn.q_proj.weight")
	pub name: String,
	/// Shape in elements per dimension
	pub shape: Dim,
	/// Total element count (product of shape)
	pub elem_count: usize,
	/// Element type
	pub dtype: GgmlType,
	/// Classification of this tensor's role
	pub kind: TensorType,
	/// Original byte offset in the source file
	pub src_offset: u64,
	/// Size in bytes in the source file (before any compression)
	pub src_size: u64,
	/// Target byte offset in the sandbag output (set during conversion)
	pub dest_offset: Option<u64>,
}

impl Tensor {
	pub fn new(
		name: String,
		shape: Dim,
		dtype: GgmlType,
		kind: TensorType,
		src_offset: u64,
		src_size: u64,
	) -> Self {
		let elem_count = shape.iter().product();
		Self {
			name,
			shape,
			elem_count,
			dtype,
			kind,
			src_offset,
			src_size,
			dest_offset: None,
		}
	}

	/// Return the tensor's name.
	pub fn get_tensor_name(&self) -> &str {
		&self.name
	}

	/// Return the shape (dimensions).
	pub fn get_tensor_dim(&self) -> &[usize] {
		&self.shape
	}

	/// Return (name, shape, elem_count, dtype, kind) as a tuple.
	pub fn get_tensor_members(&self) -> (&str, &[usize], usize, GgmlType, &TensorType) {
		(
			&self.name,
			&self.shape,
			self.elem_count,
			self.dtype,
			&self.kind,
		)
	}

	/// Classify a tensor by its source name.
	pub fn classify(name: &str) -> TensorType {
		let lower = name.to_lowercase();
		// Attention
		if lower.contains("q_proj") || lower.contains("query") {
			return TensorType::QKVWeight;
		}
		if lower.contains("k_proj") || lower.contains("key") {
			return TensorType::QKVWeight;
		}
		if lower.contains("v_proj") || lower.contains("value") {
			return TensorType::QKVWeight;
		}
		if lower.contains("attn") && lower.contains("out") {
			return TensorType::AttnOutWeight;
		}
		// FF
		if lower.contains("gate_proj") {
			return TensorType::GateWeight;
		}
		if lower.contains("up_proj") {
			return TensorType::UpWeight;
		}
		if lower.contains("down_proj") {
			return TensorType::DownWeight;
		}
		// Norm
		if lower.contains("rms_norm") || lower.contains("layernorm") {
			if lower.contains("weight") || lower.contains("gamma") {
				return TensorType::RmsNormWeight;
			}
			if lower.contains("bias") || lower.contains("beta") {
				return TensorType::RmsNormBias;
			}
		}
		// Output / LM head
		if lower.contains("lm_head") || lower.contains("output") || lower.contains("embed_tokens") {
			return TensorType::OutputWeight;
		}
		// Embedding
		if lower.contains("embedding") || lower.contains("tok_embeddings") {
			return TensorType::EmbeddingWeight;
		}
		// Rope
		if lower.contains("rope") || lower.contains("freq") {
			return TensorType::RopeFreqs;
		}
		// Bias fallback
		if lower.ends_with("_bias") || lower.contains("bias") {
			if let Some(layer_num) = Self::extract_layer(&lower) {
				return TensorType::Bias { layer: layer_num };
			}
			return TensorType::Unknown;
		}
		// Generic weight with layer
		if let Some(layer_num) = Self::extract_layer(&lower) {
			return TensorType::Weight {
				layer: layer_num,
				kind: lower.split('.').last().unwrap_or("unknown").to_string(),
			};
		}
		TensorType::Unknown
	}

	fn extract_layer(name: &str) -> Option<usize> {
		let mut parts = name.split('.');
		let mut layer: Option<usize> = None;
		while let Some(part) = parts.next() {
			if part == "layers" {
				if let Some(next) = parts.next() {
					layer = next.parse::<usize>().ok();
					break;
				}
			}
		}
		layer
	}
}

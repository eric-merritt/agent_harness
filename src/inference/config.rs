// Model configuration — supports both Qwen3.5 (GGUF) and Qwen2 (HuggingFace) formats.

use crate::models::formats::gguf::{GGUFFile, GGUFValue};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
	#[serde(default, alias = "num_hidden_layers")]
	pub n_layer: usize,
	#[serde(default)]
	pub n_layer_nextn: usize,
	#[serde(default, alias = "hidden_size")]
	pub n_embd: usize,
	#[serde(default, alias = "intermediate_size")]
	pub n_ff: usize,
	#[serde(default, alias = "num_attention_heads")]
	pub n_head: usize,
	#[serde(default, alias = "num_key_value_heads")]
	pub n_head_kv: usize,
	#[serde(default, alias = "head_dim")]
	pub n_embd_head: usize,
	#[serde(default)]
	pub rope_dim_count: usize,
	#[serde(default)]
	pub rope_sections: [i32; 4],
	#[serde(default, alias = "rope_theta")]
	pub rope_freq_base: f32,
	#[serde(default, alias = "rms_norm_eps")]
	pub rms_eps: f32,
	#[serde(default)]
	pub ssm_d_conv: usize,
	#[serde(default)]
	pub ssm_d_inner: usize,
	#[serde(default)]
	pub ssm_d_state: usize,
	#[serde(default)]
	pub ssm_dt_rank: usize,
	#[serde(default)]
	pub ssm_n_group: usize,
	#[serde(default)]
	pub full_attn_interval: usize,
	#[serde(default, alias = "max_position_embeddings")]
	pub context_length: usize,
	#[serde(default)]
	pub vocab_size: usize,
	#[serde(default)]
	pub eos_token_id: u32,
	#[serde(default)]
	pub pad_token_id: u32,
	#[serde(skip)] // computed, not deserialized
	pub is_recurrent: Vec<bool>,
	/// Per-layer type from config: "linear_attention", "full_attention", "sliding_attention", etc.
	/// Empty if config doesn't specify (e.g. Qwen2).
	#[serde(default, alias = "layer_types")]
	pub layer_types: Vec<String>,
	#[serde(default, alias = "max_position_embeddings")]
	pub max_seq_len: usize,
}

impl ModelConfig {
	pub fn from_gguf(gguf: &GGUFFile) -> Self {
		let get_u32 = |key: &str| -> u32 {
			gguf.kv_meta
				.get(key)
				.and_then(|v| match v {
					GGUFValue::U32(v) => Some(*v),
					GGUFValue::U64(v) => Some(*v as u32),
					_ => None,
				})
				.unwrap_or(0)
		};
		let get_u64 = |key: &str| -> u64 {
			gguf.kv_meta
				.get(key)
				.and_then(|v| match v {
					GGUFValue::U64(v) => Some(*v),
					GGUFValue::U32(v) => Some(*v as u64),
					_ => None,
				})
				.unwrap_or(0)
		};
		let get_f32 = |key: &str| -> f32 {
			gguf.kv_meta
				.get(key)
				.and_then(|v| match v {
					GGUFValue::F32(v) => Some(*v),
					_ => None,
				})
				.unwrap_or(0.0)
		};

		// Detect architecture prefix from GGUF metadata
		let prefix = gguf
			.kv_meta
			.get("general.architecture")
			.and_then(|v| match v {
				GGUFValue::String(s) => Some(s.clone()),
				_ => None,
			})
			.unwrap_or_else(|| "qwen35".to_string());
		let prefix = format!("{}.", prefix);
		let n_layer = get_u64(&format!("{}block_count", prefix)) as usize;
		let n_embd = get_u64(&format!("{}embedding_length", prefix)) as usize;
		let n_ff = get_u64(&format!("{}feed_forward_length", prefix)) as usize;
		let n_head = get_u64(&format!("{}attention.head_count", prefix)) as usize;
		let n_head_kv = get_u64(&format!("{}attention.head_count_kv", prefix)) as usize;
		let n_embd_head = get_u32(&format!("{}attention.key_length", prefix)) as usize;
		let rope_dim_count = get_u64(&format!("{}rope.dimension_count", prefix)) as usize;
		let rms_eps = get_f32(&format!("{}attention.layer_norm_rms_epsilon", prefix));
		let rope_freq_base = get_f32(&format!("{}rope.freq_base", prefix));
		let ssm_d_conv = get_u64(&format!("{}ssm.conv_kernel", prefix)) as usize;
		let ssm_d_inner = get_u64(&format!("{}ssm.inner_size", prefix)) as usize;
		let ssm_d_state = get_u64(&format!("{}ssm.state_size", prefix)) as usize;
		let ssm_dt_rank = get_u64(&format!("{}ssm.time_step_rank", prefix)) as usize;
		let ssm_n_group = get_u64(&format!("{}ssm.group_count", prefix)) as usize;
		let full_attn_interval = get_u32(&format!("{}full_attention_interval", prefix)) as usize;
		let n_layer_nextn = get_u64(&format!("{}nextn_predict_layers", prefix)) as usize;
		let context_length = get_u64(&format!("{}context_length", prefix)) as usize;

		let eos_token_id = get_u32("tokenizer.ggml.eos_token_id");
		let pad_token_id = get_u32("tokenizer.ggml.padding_token_id");

		let vocab_size = gguf
			.tensor_info
			.iter()
			.find(|t| t.name == "token_embd.weight")
			.map(|t| t.dim[1] as usize)
			.unwrap_or(0);

		let rope_sections: [i32; 4] = gguf
			.kv_meta
			.get(&format!("{}rope.dimension_sections", prefix))
			.and_then(|v| match v {
				GGUFValue::Array(arr) => {
					let vals: Vec<i32> = arr
						.data
						.iter()
						.filter_map(|x| match x {
							GGUFValue::U64(v) => Some(*v as i32),
							GGUFValue::U32(v) => Some(*v as i32),
							_ => None,
						})
						.collect();
					vals.try_into().ok().map(|a: [i32; 4]| a)
				}
				_ => None,
			})
			.unwrap_or([0, 0, 0, 0]);

		let n_layer_all = n_layer + n_layer_nextn;
		let is_recurrent = (0..n_layer_all)
			.map(|i| {
				if i >= n_layer {
					return false;
				}
				if i == n_layer - 1 {
					return false;
				}
				(i + 1) % full_attn_interval.max(1) != 0
			})
			.collect();

		// Safe resolution baseline for GGUF contexts
		let max_seq_len = if context_length > 0 {
			context_length
		} else {
			4096
		};

		Self {
			n_layer,
			n_layer_nextn,
			n_embd,
			n_ff,
			n_head,
			n_head_kv,
			n_embd_head,
			rope_dim_count,
			rope_sections,
			rope_freq_base,
			rms_eps,
			ssm_d_conv,
			ssm_d_inner,
			ssm_d_state,
			ssm_dt_rank,
			ssm_n_group,
			full_attn_interval,
			vocab_size,
			context_length,
			eos_token_id,
			pad_token_id,
			is_recurrent,
			layer_types: Vec::new(),
			max_seq_len,
		}
	}

	/// Load config from a model directory (reads config.json).
	/// Handles both HuggingFace (Qwen2) and GGUF-style (Qwen3.5) config formats.
	pub fn from_dir(dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
		let path = dir.join("config.json");
		let data = std::fs::read_to_string(&path)?;
		let mut config: ModelConfig = serde_json::from_str(&data)?;

		// Compute derived fields that can't be deserialized directly

		// n_embd_head: if not in config, compute from n_embd / n_head
		if config.n_embd_head == 0 && config.n_head > 0 {
			config.n_embd_head = config.n_embd / config.n_head;
		}

		// rope_dim_count: if not set, use n_embd_head (standard RoPE on all head dims)
		if config.rope_dim_count == 0 {
			config.rope_dim_count = config.n_embd_head;
		}

		// rope_sections: if all zero (Qwen2 standard RoPE), use single section
		if config.rope_sections == [0, 0, 0, 0] {
			config.rope_sections = [config.rope_dim_count as i32 / 2, 0, 0, 0];
		}

		// full_attn_interval: default 1 (all attention) for non-SSM architectures
		if config.full_attn_interval == 0 {
			config.full_attn_interval = 1;
		}

		// Fallback baseline for max_seq_len field initialization if absent from direct JSON mappings
		if config.max_seq_len == 0 {
			config.max_seq_len = if config.context_length > 0 {
				config.context_length
			} else {
				4096
			};
		}

		// is_recurrent: if config has SSM fields or layer_types designations, build the recurrent mask.
		// Otherwise (Qwen2), all layers are attention (all false).
		let has_ssm = config.ssm_d_state > 0;
		let n_layer_all = config.n_layer + config.n_layer_nextn;

		if has_ssm || !config.layer_types.is_empty() {
			config.is_recurrent = (0..n_layer_all)
				.map(|i| {
					if i >= config.n_layer {
						return false;
					}
					if let Some(l_type) = config.layer_types.get(i) {
						l_type.contains("recurrent") || l_type.contains("linear_attention")
					} else {
						if i == config.n_layer - 1 {
							return false;
						}
						(i + 1) % config.full_attn_interval.max(1) != 0
					}
				})
				.collect();
		} else {
			config.is_recurrent = vec![false; n_layer_all];
		}

		Ok(config)
	}

	pub fn to_file(&self, dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
		let path = dir.join("config.json");
		let data = serde_json::to_string_pretty(self)?;
		std::fs::write(&path, data)?;
		Ok(())
	}
}

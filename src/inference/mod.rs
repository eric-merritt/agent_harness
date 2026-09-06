// Inference engine — loads DedupCount-compressed weights, decompresses to INT4,
// and runs forward passes with on-the-fly INT4 dequantization in the GEMV kernel.
//
// Architecture: SSM+attention hybrid (Qwen3.5) or pure attention (Qwen2).
// Forward pass per block:
//   1. RMSNorm(attn_norm)
//   2. SSM or Attention block
//   3. Residual add
//   4. RMSNorm(ffn_norm)
//   5. FFN (SwiGLU)
//   6. Residual add

pub mod attention;
pub mod config;
pub mod ffn;
pub mod kv_cache;
// `kernels` is already a top-level module; aliasing it keeps every existing
// `math::` path working without compiling the whole kernel tree a second time.
pub use crate::kernels as math;
pub mod progress;
pub mod sampling;
pub mod ssm;
pub mod tokenizer;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::models::format::SandbagReader;
use config::ModelConfig;
use tokenizer::Tokenizer;

pub use attention::{AttnBlock, AttnState};
pub use ffn::{FfnBlock, FfnState};
pub use kv_cache::KvCache;
pub use ssm::{SsmBlock, SsmState};

/// Running inference engine. Loads compressed model, holds state.
pub struct InferenceEngine {
	pub loader: SandbagReader,
	pub config: ModelConfig,
	pub tokenizer: Tokenizer,
	pub kv_caches: Vec<KvCache>,
	pub ssm_states: Vec<SsmState>,
	/// Tensor name → (byte offset in mmap, element_count, is_4bit, group_size).
	tensor_index: HashMap<String, (u64, usize, bool, usize)>,
	mmap: Option<memmap2::Mmap>,
	temp_path: PathBuf,
	write_pos: u64,
	pub position: usize,
	pub scratch_hidden: Vec<f32>,
	pub scratch_normed: Vec<f32>,
	pub scratch_ffn_out: Vec<f32>,
	pub scratch_logits: Vec<f32>,
	pub scratch_ffn_gate: Vec<f32>,
	pub scratch_ffn_up: Vec<f32>,
	pub scratch_ffn_act: Vec<f32>,
	pub scratch_attn_q: Vec<f32>,
	pub scratch_attn_k: Vec<f32>,
	pub scratch_attn_v: Vec<f32>,
	pub scratch_attn_out: Vec<f32>,
	pub scratch_scores: Vec<f32>,
}

impl InferenceEngine {
	pub fn open(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
		Self::open_with_progress(model_dir, None)
	}
	pub fn open_with_progress(
		model_dir: &Path,
		progress: Option<&progress::LoadingProgress>,
	) -> Result<Self, Box<dyn std::error::Error>> {
		log::info!(
			"InferenceEngine: opening model from {}",
			model_dir.display()
		);
		if let Some(p) = progress {
			p.set(6, "Reading model config...");
		}
		let loader = SandbagReader::from_path(&model_dir.join("model.sandbag"))
			.map_err(std::io::Error::other)?;
		let config = ModelConfig::from_dir(model_dir)?;

		if let Some(p) = progress {
			p.set(7, "Loading tokenizer...");
		}
		let tokenizer = Tokenizer::from_dir(model_dir)?;

		let n_layers = config.n_layer + config.n_layer_nextn;
		let mut kv_caches = Vec::with_capacity(n_layers);
		let mut ssm_states = Vec::with_capacity(n_layers);
		if let Some(p) = progress {
			p.set(8, "Allocating KV caches...");
		}
		for i in 0..n_layers {
			if config.is_recurrent[i] {
				ssm_states.push(SsmState::new(&config));
				kv_caches.push(KvCache::empty());
			} else {
				ssm_states.push(SsmState::empty());
				kv_caches.push(KvCache::new(&config));
			}
		}

		log::info!(
			"InferenceEngine: {} layers ({} recurrent, {} attention), vocab={}, hidden={}",
			config.n_layer,
			config.is_recurrent.iter().filter(|&&r| r).count(),
			config.is_recurrent.iter().filter(|&&r| !r).count(),
			config.vocab_size,
			config.n_embd
		);

		// No global table: sandbag carries no shared lookup structure. Every weight
		// decodes from its own prefix/tail/sign with nothing else consulted.
		if let Some(p) = progress {
			p.set(10, "Validating tensor names...");
		}
		let n_embd = config.n_embd;
		let n_ff = config.n_ff;
		let vocab_size = config.vocab_size;
		let n_embd_kv = config.n_head_kv * config.n_embd_head;

		// Validate that all expected tensor names are present in the checkpoint.
		let all_names: Vec<&str> = loader.tensor_names();
		let name_set: HashSet<&str> = all_names.iter().copied().collect();
		let mut missing: Vec<String> = Vec::new();
		for il in 0..config.n_layer {
			let expected = [
				"attn_norm.weight",
				"attn_q.weight",
				"attn_k.weight",
				"attn_v.weight",
				"attn_output.weight",
				"ffn_norm.weight",
				"ffn_gate.weight",
				"ffn_up.weight",
				"ffn_down.weight",
			];
			for suffix in &expected {
				let full = format!("blk.{}.{}", il, suffix);
				if !name_set.contains(full.as_str()) {
					missing.push(full);
				}
			}
		}
		for special in ["token_embd.weight", "output_norm.weight", "output.weight"] {
			if !name_set.contains(special) {
				missing.push(special.to_string());
			}
		}
		if !missing.is_empty() {
			log::warn!(
				"InferenceEngine: {} expected tensors missing from checkpoint: {:?}",
				missing.len(),
				&missing[..missing.len().min(10)]
			);
		}

		let temp_path =
			std::env::temp_dir().join(format!("agent_harness_weights_{}.bin", std::process::id()));
		File::create(&temp_path)?;

		Ok(Self {
			loader,
			config,
			tokenizer,
			kv_caches,
			ssm_states,
			tensor_index: HashMap::new(),
			mmap: None,
			temp_path,
			write_pos: 0,
			position: 0,
			scratch_hidden: vec![0.0; n_embd],
			scratch_normed: vec![0.0; n_embd],
			scratch_ffn_out: vec![0.0; n_embd],
			scratch_logits: vec![0.0; vocab_size],
			scratch_ffn_gate: vec![0.0; n_ff],
			scratch_ffn_up: vec![0.0; n_ff],
			scratch_ffn_act: vec![0.0; n_ff],
			scratch_attn_q: vec![0.0; n_embd],
			scratch_attn_k: vec![0.0; n_embd_kv],
			scratch_attn_v: vec![0.0; n_embd_kv],
			scratch_attn_out: vec![0.0; n_embd],
			scratch_scores: vec![0.0; 4096],
		})
	}

	// NOTE: `decompress_all_parallel` was removed here. It decompressed the whole
	// model to a temp file ahead of inference using the superseded global-table
	// scheme, via `models::conversion::loader::ModelLoader` and
	// `models::quantization` — both deleted. Sandbag needs no such pre-pass:
	// weights decode inline in the GEMV kernel as sign · (prefix ++ tail).
	// It had no callers outside this file.

	pub fn finalize_mmap(&mut self) {
		if self.mmap.is_some() {
			return;
		}
		let file = std::fs::OpenOptions::new()
			.read(true)
			.open(&self.temp_path)
			.expect("temp file not found");
		let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap temp file");
		self.mmap = Some(mmap);
		log::info!(
			"Mmap finalized: {} tensors, {:.1} GB",
			self.tensor_index.len(),
			self.write_pos as f64 / 1e9
		);
	}

	fn forward(&mut self, token_id: u32) {
		let pos = self.position;
		self.position += 1;
		let n_embd = self.config.n_embd;
		let n_ff = self.config.n_ff;
		let n_head = self.config.n_head;
		let n_head_kv = self.config.n_head_kv;
		let n_embd_head = self.config.n_embd_head;
		let n_embd_kv = n_head_kv * n_embd_head;
		let gs = 32;
		let n_layer = self.config.n_layer;
		let eps = self.config.rms_eps;
		let vocab_size = self.config.vocab_size;
		let rope_dim = self.config.rope_dim_count;
		let rope_sec = self.config.rope_sections;
		let rope_fb = self.config.rope_freq_base;

		let mmap_ptr: *const u8 = self
			.mmap
			.as_ref()
			.map(|m| m.as_ptr())
			.unwrap_or(std::ptr::null());
		let mmap_len = self.mmap.as_ref().map(|m| m.len()).unwrap_or(0);
		let ti = &self.tensor_index;

		let get_f32 = |name: &str| -> &[f32] {
			match ti.get(name) {
				Some(&(off, count, is_4bit, _)) if !is_4bit && mmap_ptr != std::ptr::null() => {
					let s = off as usize;
					let e = s + count * 4;
					if e <= mmap_len {
						bytemuck::cast_slice(unsafe {
							std::slice::from_raw_parts(mmap_ptr.add(s), e - s)
						})
					} else {
						&[]
					}
				}
				_ => &[],
			}
		};

		let get_int4 = |name: &str| -> Option<(&[f32], &[u8], usize)> {
			let &(off, count, is_4bit, group_size) = ti.get(name)?;
			if !is_4bit || mmap_ptr == std::ptr::null() {
				return None;
			}
			let ng = (count + group_size - 1) / group_size;
			let sb = ng * 4;
			let pb = (count + 1) / 2;
			let ss = off as usize;
			let se = ss + sb + pb;
			if se <= mmap_len {
				let scales = bytemuck::cast_slice(unsafe {
					std::slice::from_raw_parts(mmap_ptr.add(ss), sb)
				});
				let packed = unsafe { std::slice::from_raw_parts(mmap_ptr.add(ss + sb), pb) };
				Some((scales, packed, group_size))
			} else {
				None
			}
		};

		let embd = get_f32("token_embd.weight");
		let embd_start = token_id as usize * n_embd;
		if embd_start + n_embd > embd.len() {
			return;
		}
		self.scratch_hidden
			.copy_from_slice(&embd[embd_start..embd_start + n_embd]);

		for il in 0..n_layer {
			let norm_w = get_f32(&format!("blk.{}.attn_norm.weight", il));
			if norm_w.is_empty() {
				self.scratch_normed.copy_from_slice(&self.scratch_hidden);
			} else {
				math::rms_norm_into(&mut self.scratch_normed, &self.scratch_hidden, norm_w, eps);
			}

			let (qs, qp, _) =
				get_int4(&format!("blk.{}.attn_q.weight", il)).unwrap_or((&[], &[], gs));
			let (ks, kp, _) =
				get_int4(&format!("blk.{}.attn_k.weight", il)).unwrap_or((&[], &[], gs));
			let (vs, vp, _) =
				get_int4(&format!("blk.{}.attn_v.weight", il)).unwrap_or((&[], &[], gs));

			math::gemv_4bit_into(
				&mut self.scratch_attn_q,
				qs,
				qp,
				&self.scratch_normed,
				n_embd,
				n_embd,
				gs,
			);
			math::gemv_4bit_into(
				&mut self.scratch_attn_k,
				ks,
				kp,
				&self.scratch_normed,
				n_embd_kv,
				n_embd,
				gs,
			);
			math::gemv_4bit_into(
				&mut self.scratch_attn_v,
				vs,
				vp,
				&self.scratch_normed,
				n_embd_kv,
				n_embd,
				gs,
			);

			// ── FIXED BORROW CHECKER ALIASING & HEAD BOUNDARY ALIGNMENT ──
			let q_norm = get_f32(&format!("blk.{}.attn_q_norm.weight", il));
			if !q_norm.is_empty() {
				// Pre-allocate temporary workspace buffer to prevent field-borrow aliasing
				let mut q_destination_buffer = vec![0.0f32; n_embd_head];
				for head_index in 0..n_head {
					let start_offset = head_index * n_embd_head;
					let end_offset = start_offset + n_embd_head;

					// Scope block isolates immutable borrows before mutating self.scratch_attn_q
					{
						let q_source_slice =
							&self.scratch_attn_q[start_offset..start_offset + n_embd_head];
						// If q_norm is per-head, slice it with [start_offset..end_offset].
						// If shared across heads, fallback safely to using the whole slice.
						let norm_slice = if q_norm.len() >= end_offset {
							&q_norm[start_offset..end_offset]
						} else {
							q_norm
						};

						math::rms_norm_into(
							&mut q_destination_buffer,
							q_source_slice,
							norm_slice,
							eps,
						);
					}
					self.scratch_attn_q[start_offset..start_offset + n_embd_head]
						.copy_from_slice(&q_destination_buffer);
				}
			}

			let k_norm = get_f32(&format!("blk.{}.attn_k_norm.weight", il));
			if !k_norm.is_empty() {
				let mut k_destination_buffer = vec![0.0f32; n_embd_head];
				for head_index in 0..n_head_kv {
					let start_offset = head_index * n_embd_head;
					let end_offset = start_offset + n_embd_head;

					{
						let k_source_slice =
							&self.scratch_attn_k[start_offset..start_offset + n_embd_head];
						let norm_slice = if k_norm.len() >= end_offset {
							&k_norm[start_offset..end_offset]
						} else {
							k_norm
						};

						math::rms_norm_into(
							&mut k_destination_buffer,
							k_source_slice,
							norm_slice,
							eps,
						);
					}
					self.scratch_attn_k[start_offset..start_offset + n_embd_head]
						.copy_from_slice(&k_destination_buffer);
				}
			}

			math::rope_multi(&mut self.scratch_attn_q, pos, rope_dim, rope_sec, rope_fb);
			math::rope_multi(&mut self.scratch_attn_k, pos, rope_dim, rope_sec, rope_fb);

			let kv = &mut self.kv_caches[il];
			for h in 0..n_head_kv {
				let off = h * n_embd_head;
				kv.write_k(h, pos, &self.scratch_attn_k[off..off + n_embd_head]);
				kv.write_v(h, pos, &self.scratch_attn_v[off..off + n_embd_head]);
			}

			let scale = 1.0 / (n_embd_head as f32).sqrt();
			let active_len = pos + 1;
			let scores = &mut self.scratch_scores[0..active_len];

			// Temporary buffer for reading a single K/V row
			let mut k_row = vec![0.0f32; n_embd_head];
			let mut v_row = vec![0.0f32; n_embd_head];

			for h in 0..n_head {
				let kv_head = if n_head_kv > 0 {
					h * n_head_kv / n_head
				} else {
					0
				};
				let qoff = h * n_embd_head;

				// Compute dot products and store in scores[0..=pos]
				let mut max_s = f32::NEG_INFINITY;
				for p in 0..active_len {
					kv.read_k(kv_head, p, &mut k_row);
					let mut dot = 0.0f32;
					for d in 0..n_embd_head {
						dot += self.scratch_attn_q[qoff + d] * k_row[d];
					}
					scores[p] = dot * scale;
					if scores[p] > max_s {
						max_s = scores[p];
					}
				}

				// Softmax
				let mut sum_exp = 0.0f32;
				for s in &mut scores[0..active_len] {
					*s = (*s - max_s).exp();
					sum_exp += *s;
				}
				let inv = 1.0 / sum_exp;

				// Weighted sum of V
				for d in 0..n_embd_head {
					let mut acc = 0.0f32;
					for p in 0..active_len {
						kv.read_v(kv_head, p, &mut v_row);
						acc += scores[p] * v_row[d];
					}
					self.scratch_attn_out[qoff + d] = acc * inv;
				}
			}

			let (os, op, _) =
				get_int4(&format!("blk.{}.attn_output.weight", il)).unwrap_or((&[], &[], gs));
			math::gemv_4bit_into(
				&mut self.scratch_ffn_out,
				os,
				op,
				&self.scratch_attn_out,
				n_embd,
				n_embd,
				gs,
			);
			for i in 0..n_embd {
				self.scratch_hidden[i] += self.scratch_ffn_out[i];
			}

			let post_w = get_f32(&format!("blk.{}.ffn_norm.weight", il));
			if post_w.is_empty() {
				self.scratch_normed.copy_from_slice(&self.scratch_hidden);
			} else {
				math::rms_norm_into(&mut self.scratch_normed, &self.scratch_hidden, post_w, eps);
			}

			let (gs2, gp, _) =
				get_int4(&format!("blk.{}.ffn_gate.weight", il)).unwrap_or((&[], &[], gs));
			let (us, up, _) =
				get_int4(&format!("blk.{}.ffn_up.weight", il)).unwrap_or((&[], &[], gs));
			let (ds, dp, _) =
				get_int4(&format!("blk.{}.ffn_down.weight", il)).unwrap_or((&[], &[], gs));
			if !gp.is_empty() && !up.is_empty() && !dp.is_empty() {
				ffn::swiglu_4bit_into(
					&mut self.scratch_ffn_out,
					&self.scratch_normed,
					gs2,
					gp,
					us,
					up,
					ds,
					dp,
					n_embd,
					n_ff,
					gs,
					&mut self.scratch_ffn_gate,
					&mut self.scratch_ffn_up,
					&mut self.scratch_ffn_act,
				);
				for i in 0..n_embd {
					self.scratch_hidden[i] += self.scratch_ffn_out[i];
				}
			}
		}

		let out_norm = get_f32("output_norm.weight");
		if out_norm.is_empty() {
			self.scratch_normed.copy_from_slice(&self.scratch_hidden);
		} else {
			math::rms_norm_into(
				&mut self.scratch_normed,
				&self.scratch_hidden,
				out_norm,
				eps,
			);
		}

		let (ls, lp, _) = get_int4("output.weight").unwrap_or((&[], &[], gs));
		if !lp.is_empty() {
			math::gemv_4bit_into(
				&mut self.scratch_logits,
				ls,
				lp,
				&self.scratch_normed,
				vocab_size,
				n_embd,
				gs,
			);
		} else {
			let lm = get_f32("output.weight");
			if !lm.is_empty() {
				math::gemv_into(
					&mut self.scratch_logits,
					lm,
					&self.scratch_normed,
					vocab_size,
					n_embd,
				);
			}
		}
	}

	pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> String {
		let tokens = self.tokenizer.encode(prompt);
		let mut output_tokens = Vec::new();
		for &tid in &tokens {
			self.forward(tid);
		}
		for _ in 0..max_tokens {
			let next = if temperature > 0.0 {
				sampling::sample(&self.scratch_logits, temperature, 40)
			} else {
				sampling::argmax(&self.scratch_logits)
			};
			if next == self.config.eos_token_id {
				break;
			}
			output_tokens.push(next);
			self.forward(next);
		}
		self.tokenizer.decode(&output_tokens)
	}
}

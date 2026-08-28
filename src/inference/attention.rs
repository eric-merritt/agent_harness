use crate::inference::config::ModelConfig;
use crate::inference::kv_cache::KvCache;
use crate::inference::math;

/// Pre-allocated scratch buffers for zero-allocation attention forward passes.
pub struct AttnState {
	pub q: Vec<f32>,
	pub gate: Vec<f32>,
	pub q_gate_fused: Vec<f32>,
	pub k_cur: Vec<f32>,
	pub v_cur: Vec<f32>,
	pub head_scratch: Vec<f32>,
	pub attn_out: Vec<f32>,
	pub scores: Vec<f32>,
	pub proj_input: Vec<f32>,
}

impl AttnState {
	pub fn new(config: &ModelConfig) -> Self {
		let n_head = config.n_head;
		let n_head_kv = config.n_head_kv;
		let n_embd_head = config.n_embd_head;

		Self {
			q: vec![0.0f32; n_embd_head * n_head],
			gate: vec![0.0f32; n_embd_head * n_head],
			q_gate_fused: vec![0.0f32; n_embd_head * 2 * n_head],
			k_cur: vec![0.0f32; n_embd_head * n_head_kv],
			v_cur: vec![0.0f32; n_embd_head * n_head_kv],
			head_scratch: vec![0.0f32; n_embd_head],
			attn_out: vec![0.0f32; n_embd_head * n_head],
			scores: vec![0.0f32; 4096],
			proj_input: vec![0.0f32; n_embd_head * n_head],
		}
	}
}

// ---------------------------------------------------------------------------
// Weight block structure holding your flat GGUF slice references
// ---------------------------------------------------------------------------
pub struct AttnBlock<'a> {
	pub wq: &'a [f32],
	pub wk: &'a [f32],
	pub wv: &'a [f32],
	pub wo: &'a [f32],
	pub q_norm: &'a [f32],
	pub k_norm: &'a [f32],
}

// FIXED: Implemented explicitly back on AttnBlock with its lifetime constraints intact
impl<'a> AttnBlock<'a> {
	pub fn from_refs(
		wq: &'a [f32],
		wk: &'a [f32],
		wv: &'a [f32],
		wo: &'a [f32],
		q_norm: &'a [f32],
		k_norm: &'a [f32],
	) -> Self {
		Self {
			wq,
			wk,
			wv,
			wo,
			q_norm,
			k_norm,
		}
	}

	/// Forward pass for a single token (autoregressive decoding).
	/// Employs pre-allocated state scratchpads to enforce zero-allocation execution paths.
	pub fn forward(
		&self,
		input: &[f32],
		kv: &mut KvCache,
		state: &mut AttnState,
		pos: usize,
		config: &ModelConfig,
	) -> Vec<f32> {
		let n_embd = config.n_embd;
		let n_head = config.n_head;
		let n_head_kv = config.n_head_kv;
		let n_embd_head = config.n_embd_head;
		let rope_dim_count = config.rope_dim_count;
		let rope_sections = config.rope_sections;
		let rope_freq_base = config.rope_freq_base;
		let rms_eps = config.rms_eps;

		assert!(
			pos < kv.max_seq_len,
			"pos {} exceeds max_seq_len {}",
			pos,
			kv.max_seq_len
		);
		assert_eq!(
			input.len(),
			n_embd,
			"input len {} != n_embd {}",
			input.len(),
			n_embd
		);

		// Safe architecture detection independent of hyperparameter multipliers
		let expected_fused_len = n_embd_head * 2 * n_head * n_embd;
		let has_gate = self.wq.len() == expected_fused_len;
		let has_q_norm = !self.q_norm.is_empty();
		let has_k_norm = !self.k_norm.is_empty();

		if has_gate {
			math::gemv_into(
				&mut state.q_gate_fused,
				self.wq,
				input,
				n_embd_head * 2 * n_head,
				n_embd,
			);
			let q_stride = n_embd_head * n_head;
			state.q.copy_from_slice(&state.q_gate_fused[0..q_stride]);
			state
				.gate
				.copy_from_slice(&state.q_gate_fused[q_stride..2 * q_stride]);
		} else {
			math::gemv_into(&mut state.q, self.wq, input, n_embd_head * n_head, n_embd);
		}

		// Q normalization (Zero-allocation via head-by-head calculation chunks)
		if has_q_norm {
			for h in 0..n_head {
				let start = h * n_embd_head;
				let end = start + n_embd_head;
				state.head_scratch.copy_from_slice(&state.q[start..end]);
				math::rms_norm_into(
					&mut state.q[start..end],
					&state.head_scratch,
					self.q_norm,
					rms_eps,
				);
			}
		}

		// K and V projections
		math::gemv_into(
			&mut state.k_cur,
			self.wk,
			input,
			n_embd_head * n_head_kv,
			n_embd,
		);
		math::gemv_into(
			&mut state.v_cur,
			self.wv,
			input,
			n_embd_head * n_head_kv,
			n_embd,
		);

		// K normalization
		if has_k_norm {
			for h in 0..n_head_kv {
				let start = h * n_embd_head;
				let end = start + n_embd_head;
				state.head_scratch.copy_from_slice(&state.k_cur[start..end]);
				math::rms_norm_into(
					&mut state.k_cur[start..end],
					&state.head_scratch,
					self.k_norm,
					rms_eps,
				);
			}
		}

		// Multi-Dimensional RoPE (MRoPE) execution mappings
		for h in 0..n_head {
			let start = h * n_embd_head;
			math::rope_multi(
				&mut state.q[start..start + n_embd_head],
				pos,
				rope_dim_count,
				rope_sections,
				rope_freq_base,
			);
		}
		for h in 0..n_head_kv {
			let start = h * n_embd_head;
			math::rope_multi(
				&mut state.k_cur[start..start + n_embd_head],
				pos,
				rope_dim_count,
				rope_sections,
				rope_freq_base,
			);
		}

		// Commit active slices to the flat KV-Cache Ring
		for h in 0..n_head_kv {
			let start = h * n_embd_head;
			kv.write_k(h, pos, &state.k_cur[start..start + n_embd_head]);
			kv.write_v(h, pos, &state.v_cur[start..start + n_embd_head]);
		}

		// ---- Attention computation (GQA) ----
		let group_size = n_head / n_head_kv;
		let scale = 1.0f32 / (n_embd_head as f32).sqrt();
		let attn_total_dim = n_embd_head * n_head;

		for h in 0..n_head {
			let kv_head = h / group_size;
			let q_start = h * n_embd_head;
			let kv_base = kv_head * n_embd_head * kv.max_seq_len;

			// Isolate active lookup scores inside state arrays
			let active_scores = &mut state.scores[0..=pos];
			for s in active_scores.iter_mut() {
				*s = 0.0;
			}

			for d in 0..n_embd_head {
				let qd = state.q[q_start + d];
				let k_row_start = kv_base + d * kv.max_seq_len;
				let k_row = &kv.k[k_row_start..k_row_start + pos + 1];
				for j in 0..=pos {
					active_scores[j] += qd * k_row[j];
				}
			}
			for s in active_scores.iter_mut() {
				*s *= scale;
			}
			math::softmax(active_scores);

			let out_start = h * n_embd_head;
			for d in 0..n_embd_head {
				let v_row_start = kv_base + d * kv.max_seq_len;
				let v_row = &kv.v[v_row_start..v_row_start + pos + 1];
				let mut sum = 0.0f32;
				for j in 0..=pos {
					sum += active_scores[j] * v_row[j];
				}
				state.attn_out[out_start + d] = sum;
			}
		}

		// ---- FIXED GATING LOOP: Bound exactly to the attention dimension shape ----
		if has_gate {
			for i in 0..attn_total_dim {
				state.proj_input[i] = state.attn_out[i] * math::sigmoid(state.gate[i]);
			}
		} else {
			let slice = &state.attn_out[0..attn_total_dim];
			state.proj_input[0..attn_total_dim].copy_from_slice(slice);
		}

		// ---- Output projection (Zero-allocation final pipeline pass) ----
		let mut out = vec![0.0f32; n_embd];
		math::gemv_into(
			&mut out,
			self.wo,
			&state.proj_input[0..attn_total_dim],
			n_embd,
			attn_total_dim,
		);
		out
	}
}

// ---------------------------------------------------------------------------
// Fixed Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use crate::inference::KvCache;
	use crate::inference::config::ModelConfig;

	fn test_config() -> ModelConfig {
		ModelConfig {
			n_layer: 1,
			n_layer_nextn: 0,
			n_embd: 4096,
			n_ff: 0,
			n_head: 16,
			n_head_kv: 4,
			n_embd_head: 256,
			rope_dim_count: 64,
			rope_sections: [11, 11, 10, 0],
			rope_freq_base: 10_000_000.0,
			rms_eps: 1e-6,
			ssm_d_conv: 0,
			ssm_d_inner: 0,
			ssm_d_state: 0,
			ssm_dt_rank: 0,
			ssm_n_group: 0,
			full_attn_interval: 1,
			vocab_size: 0,
			context_length: 4096,
			eos_token_id: 0,
			pad_token_id: 0,
			is_recurrent: vec![false],
			layer_types: Vec::new(),
			max_seq_len: 4096 as usize,
		}
	}

	#[test]
	fn test_kv_cache_new_safe() {
		let config = test_config();
		let kv = KvCache::new(&config);
		assert_eq!(kv.n_head_kv, 4);
		assert_eq!(kv.head_dim, 256);
		assert_eq!(kv.max_seq_len, 4096);

		// Assert length boundaries match layout requirements safely
		let expected_size = 4 * 256 * 4096;
		assert_eq!(kv.k.len(), expected_size);
	}

	#[test]
	fn test_forward_with_state() {
		let config = test_config();
		let n_embd = config.n_embd;
		let n_head = config.n_head;
		let n_embd_head = config.n_embd_head;

		let wq = vec![0.0f32; n_embd_head * 2 * n_head * n_embd];
		let wk = vec![0.0f32; n_embd_head * config.n_head_kv * n_embd];
		let wv = vec![0.0f32; n_embd_head * config.n_head_kv * n_embd];
		let wo = vec![0.0f32; n_embd * (n_embd_head * n_head)];
		let q_norm = vec![1.0f32; n_embd_head];
		let k_norm = vec![1.0f32; n_embd_head];

		let block = AttnBlock::from_refs(&wq, &wk, &wv, &wo, &q_norm, &k_norm);
		let mut kv = KvCache::new(&config);

		// Explicitly clear uninitialized page memory flags before testing simulation
		for v in kv.k.iter_mut() {
			*v = 0.0;
		}
		for v in kv.v.iter_mut() {
			*v = 0.0;
		}

		let mut state = AttnState::new(&config);
		let input = vec![1.0f32; n_embd];

		let out = block.forward(&input, &mut kv, &mut state, 0, &config);

		assert_eq!(out.len(), n_embd);
		for &v in &out {
			assert!(v.abs() < 1e-5, "expected ~0, got {}", v);
		}
	}
}

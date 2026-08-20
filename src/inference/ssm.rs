//! Gated Delta Net (linear attention) block for Qwen3.5 SSM layers.
//!
//! Implements the recurrent state-space-style update from llama.cpp's
//! delta-net-base.cpp. Each SSM layer replaces standard attention with
//! a gated delta net that maintains a per-head [S, S] state matrix.
//!
//! Forward pass (autoregressive, single token):
//!   1. Project input -> QKV (mixed) + gate z
//!   2. Compute beta = sigmoid(beta_proj @ input)          [per v_head]
//!   3. Compute gate = softplus(alpha_proj @ input + dt_bias) * A_log
//!   4. Depthwise conv1d + SiLU on QKV
//!   5. Split into Q [head_k_dim x num_k_heads], K [same], V [head_v_dim x num_v_heads]
//!   6. L2-normalize Q and K per k_head
//!   7. Repeat Q, K from num_k_heads to num_v_heads
//!   8. Per v_head: decay state, delta = (v - S@k)*beta,
//!      update S += k (x) delta, output o = S @ q
//!   9. RMSNorm per head * SiLU(z) gating
//!  10. Output projection

use super::config::ModelConfig;
use super::math;

/// Weight references for a single SSM (gated delta net) block.
/// All weights are flat row-major f32 slices borrowed from the layer cache.
pub struct SsmBlock<'a> {
    /// [n_embd, conv_dim] -- projects hidden -> conv input (Q,K,V concatenated)
    pub wqkv: &'a [f32],
    /// [n_embd, value_dim] -- gate z
    pub wqkv_gate: &'a [f32],
    /// [ssm_d_conv, conv_dim] -- depthwise conv1d kernel
    pub ssm_conv1d: &'a [f32],
    /// [dt_rank] -- bias for alpha (time step)
    pub ssm_dt_bias: &'a [f32],
    /// [dt_rank] -- A_log (decay rate)
    pub ssm_a: &'a [f32],
    /// [n_embd, dt_rank] -- projects hidden -> dt (alpha)
    pub ssm_alpha: &'a [f32],
    /// [n_embd, dt_rank] -- projects hidden -> beta
    pub ssm_beta: &'a [f32],
    /// [head_v_dim] -- group norm weights
    pub ssm_norm: &'a [f32],
    /// [value_dim, n_embd] -- output projection
    pub ssm_out: &'a [f32],
}

/// Recurrent state for an SSM block.
/// Maintains conv1d sliding window, per-head state matrices, and scratch buffers.
pub struct SsmState {
    /// Conv1d sliding window: [(ssm_d_conv - 1) * conv_dim]
    pub conv_state: Vec<f32>,
    /// Per-head state matrices: [head_v_dim * head_v_dim * num_v_heads]
    pub ssm_state: Vec<f32>,
    // Scratch buffers — pre-allocated, zero allocation per forward pass.
    qkv_mixed: Vec<f32>,      // [conv_dim]
    z: Vec<f32>,               // [value_dim]
    beta_raw: Vec<f32>,        // [dt_rank]
    beta: Vec<f32>,            // [dt_rank]
    alpha_raw: Vec<f32>,       // [dt_rank]
    gate: Vec<f32>,            // [dt_rank]
    conv_raw: Vec<f32>,        // [conv_dim]
    conv_out: Vec<f32>,        // [conv_dim]
    q: Vec<f32>,               // [key_dim]
    k: Vec<f32>,               // [key_dim]
    q_exp: Vec<f32>,           // [head_k_dim * num_v_heads]
    k_exp: Vec<f32>,           // [head_k_dim * num_v_heads]
    output: Vec<f32>,          // [value_dim]
    result: Vec<f32>,          // [value_dim]
    head_q_scaled: Vec<f32>,   // [head_v_dim]
    head_sk: Vec<f32>,         // [head_v_dim]
    head_delta: Vec<f32>,      // [head_v_dim]
    head_o: Vec<f32>,          // [head_v_dim]
    normed_head: Vec<f32>,     // [head_v_dim] (rms_norm output per head)
    out: Vec<f32>,             // [n_embd] (final output projection)
}

impl SsmState {
    /// Allocate zeroed state for a recurrent layer.
    pub fn new(config: &ModelConfig) -> Self {
        let head_k_dim = config.ssm_d_state;
        let num_k_heads = config.ssm_n_group;
        let key_dim = head_k_dim * num_k_heads;
        let head_v_dim = config.ssm_d_inner / config.ssm_dt_rank;
        let num_v_heads = config.ssm_dt_rank;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let dt_rank = config.ssm_dt_rank;

        Self {
            conv_state: vec![0.0; (config.ssm_d_conv - 1) * conv_dim],
            ssm_state: vec![0.0; head_v_dim * head_v_dim * num_v_heads],
            qkv_mixed: vec![0.0; conv_dim],
            z: vec![0.0; value_dim],
            beta_raw: vec![0.0; dt_rank],
            beta: vec![0.0; dt_rank],
            alpha_raw: vec![0.0; dt_rank],
            gate: vec![0.0; dt_rank],
            conv_raw: vec![0.0; conv_dim],
            conv_out: vec![0.0; conv_dim],
            q: vec![0.0; key_dim],
            k: vec![0.0; key_dim],
            q_exp: vec![0.0; head_k_dim * num_v_heads],
            k_exp: vec![0.0; head_k_dim * num_v_heads],
            output: vec![0.0; value_dim],
            result: vec![0.0; value_dim],
            head_q_scaled: vec![0.0; head_v_dim],
            head_sk: vec![0.0; head_v_dim],
            head_delta: vec![0.0; head_v_dim],
            head_o: vec![0.0; head_v_dim],
            normed_head: vec![0.0; head_v_dim],
            out: vec![0.0; config.n_embd],
        }
    }

    /// Empty state for attention layers (not used, but must exist).
    pub fn empty() -> Self {
        Self {
            conv_state: Vec::new(),
            ssm_state: Vec::new(),
            qkv_mixed: Vec::new(),
            z: Vec::new(),
            beta_raw: Vec::new(),
            beta: Vec::new(),
            alpha_raw: Vec::new(),
            gate: Vec::new(),
            conv_raw: Vec::new(),
            conv_out: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            q_exp: Vec::new(),
            k_exp: Vec::new(),
            output: Vec::new(),
            result: Vec::new(),
            head_q_scaled: Vec::new(),
            head_sk: Vec::new(),
            head_delta: Vec::new(),
            head_o: Vec::new(),
            normed_head: Vec::new(),
            out: Vec::new(),
        }
    }
}
impl<'a> SsmBlock<'a> {
    /// Construct from borrowed weight slices.
    #[allow(clippy::too_many_arguments)]
    pub fn from_refs(
        wqkv: &'a [f32],
        wqkv_gate: &'a [f32],
        ssm_conv1d: &'a [f32],
        ssm_dt_bias: &'a [f32],
        ssm_a: &'a [f32],
        ssm_alpha: &'a [f32],
        ssm_beta: &'a [f32],
        ssm_norm: &'a [f32],
        ssm_out: &'a [f32],
    ) -> Self {
        Self {
            wqkv,
            wqkv_gate,
            ssm_conv1d,
            ssm_dt_bias,
            ssm_a,
            ssm_alpha,
            ssm_beta,
            ssm_norm,
            ssm_out,
        }
    }

    /// Forward pass for a single token (autoregressive).
    /// Uses pre-allocated scratch buffers in `state` — zero intermediate allocation.
    pub fn forward(&self, input: &[f32], state: &mut SsmState, config: &ModelConfig) -> Vec<f32> {
        let n_embd = config.n_embd;
        let head_k_dim = config.ssm_d_state;
        let num_k_heads = config.ssm_n_group;
        let key_dim = head_k_dim * num_k_heads;
        let head_v_dim = config.ssm_d_inner / config.ssm_dt_rank;
        let num_v_heads = config.ssm_dt_rank;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let dt_rank = config.ssm_dt_rank;
        let d_conv = config.ssm_d_conv;
        let eps = config.rms_eps;

        // Step 1: Input projections → scratch buffers
        math::gemv_into(&mut state.qkv_mixed, self.wqkv, input, conv_dim, n_embd);
        math::gemv_into(&mut state.z, self.wqkv_gate, input, value_dim, n_embd);

        // Step 2: Beta — sigmoid over alpha projection
        math::gemv_into(&mut state.beta_raw, self.ssm_beta, input, dt_rank, n_embd);
        for i in 0..dt_rank {
            state.beta[i] = math::sigmoid(state.beta_raw[i]);
        }

        // Step 3: Alpha (decay/gate) — softplus + A_log
        math::gemv_into(&mut state.alpha_raw, self.ssm_alpha, input, dt_rank, n_embd);
        for i in 0..dt_rank {
            let alpha_sp = math::softplus(state.alpha_raw[i] + self.ssm_dt_bias[i]);
            state.gate[i] = alpha_sp * self.ssm_a[i];
        }

        // Step 4: Conv1d + SiLU
        math::conv1d_depthwise_into(
            &mut state.conv_raw, &state.qkv_mixed, self.ssm_conv1d, d_conv, conv_dim, &mut state.conv_state,
        );
        for i in 0..conv_dim {
            state.conv_out[i] = math::silu(state.conv_raw[i]);
        }

        // Step 5: Split into Q, K, V
        let q_base = &state.conv_out[0..key_dim];
        let k_base = &state.conv_out[key_dim..2 * key_dim];
        let v = &state.conv_out[2 * key_dim..2 * key_dim + value_dim];

        // Step 6: L2 normalize Q and K per k_head
        state.q.copy_from_slice(q_base);
        state.k.copy_from_slice(k_base);
        for h in 0..num_k_heads {
            let s = h * head_k_dim;
            let e = s + head_k_dim;
            math::l2_norm(&mut state.q[s..e], eps);
            math::l2_norm(&mut state.k[s..e], eps);
        }
        // Step 7: Repeat Q, K from num_k_heads to num_v_heads
        let repeat = num_v_heads / num_k_heads;
        for h in 0..num_v_heads {
            let src = h / repeat;
            let ss = src * head_k_dim;
            let ds = h * head_k_dim;
            state.q_exp[ds..ds + head_k_dim].copy_from_slice(&state.q[ss..ss + head_k_dim]);
            state.k_exp[ds..ds + head_k_dim].copy_from_slice(&state.k[ss..ss + head_k_dim]);
        }

        // Safety Assert: Qwen3.5 Gated Delta Net requires matching dimensions
        assert_eq!(head_k_dim, head_v_dim, "Gated Delta Net state recurrence requires head_k_dim == head_v_dim");

        // Step 8: Gated delta net (per v_head)
        let scale = 1.0f32 / (head_k_dim as f32).sqrt();
        let ssm = &mut state.ssm_state;

        for h in 0..num_v_heads {
            let q_h = &state.q_exp[h * head_k_dim..(h + 1) * head_k_dim];
            let k_h = &state.k_exp[h * head_k_dim..(h + 1) * head_k_dim];
            let v_h = &v[h * head_v_dim..(h + 1) * head_v_dim];
            let s_off = h * head_v_dim * head_k_dim;
            let s_len = head_v_dim * head_k_dim;

            // Scale Q into head_q_scaled
            for i in 0..head_k_dim {
                state.head_q_scaled[i] = q_h[i] * scale;
            }

            // Decay: S = S * exp(gate[h])
            let decay = state.gate[h].exp();
            for s in &mut ssm[s_off..s_off + s_len] {
                *s *= decay;
            }

            // sk = S @ k → head_sk
            let s_h = &ssm[s_off..s_off + s_len];
            math::gemv_into(&mut state.head_sk, s_h, k_h, head_v_dim, head_k_dim);

            // delta = (v - sk) * beta[h] → head_delta
            for i in 0..head_v_dim {
                state.head_delta[i] = (v_h[i] - state.head_sk[i]) * state.beta[h];
            }

            // OUTER PRODUCT: S += outer(delta, k)
            let s_h_mut = &mut ssm[s_off..s_off + s_len];
            for i in 0..head_v_dim {
                let di = state.head_delta[i];
                let row_offset = i * head_k_dim;
                for j in 0..head_k_dim {
                    s_h_mut[row_offset + j] += di * k_h[j];
                }
            }

            // o = S @ q_scaled → head_o, then copy into output
            math::gemv_into(&mut state.head_o, s_h_mut, &state.head_q_scaled, head_v_dim, head_k_dim);
            state.output[h * head_v_dim..(h + 1) * head_v_dim].copy_from_slice(&state.head_o);
        }

        // Step 9: Gated normalization
        for h in 0..num_v_heads {
            let o_h = &state.output[h * head_v_dim..(h + 1) * head_v_dim];
            let z_h = &state.z[h * head_v_dim..(h + 1) * head_v_dim];
            math::rms_norm_into(&mut state.normed_head, o_h, self.ssm_norm, eps);
            for i in 0..head_v_dim {
                state.result[h * head_v_dim + i] = state.normed_head[i] * math::silu(z_h[i]);
            }
        }

        // Step 10: Output projection
        math::gemv_into(&mut state.out, self.ssm_out, &state.result, n_embd, value_dim);
        state.out.clone()
    }
}

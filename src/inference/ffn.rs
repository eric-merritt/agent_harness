use crate::inference::config::ModelConfig;
use crate::inference::math;

// ---------------------------------------------------------------------------
// FFN Memory State Pad
// ---------------------------------------------------------------------------

/// Persistent scratchpads to eliminate allocations during execution turns.
pub struct FfnState {
	pub scratch_gate: Vec<f32>,
	pub scratch_up: Vec<f32>,
	pub scratch_act: Vec<f32>,
}

impl FfnState {
	pub fn new(config: &ModelConfig) -> Self {
		let n_ff = config.n_ff;
		Self {
			scratch_gate: vec![0.0f32; n_ff],
			scratch_up: vec![0.0f32; n_ff],
			scratch_act: vec![0.0f32; n_ff],
		}
	}
}

// ---------------------------------------------------------------------------
// GGUF Linear Projection Weights Struct
// ---------------------------------------------------------------------------

pub struct FfnBlock<'a> {
	// Standard floating-point view references
	pub gate_w: &'a [f32],
	pub up_w: &'a [f32],
	pub down_w: &'a [f32],

	// Quantized INT4 pointers for DedupCount formats
	pub gate_packed: &'a [u8],
	pub gate_scales: &'a [f32],
	pub up_packed: &'a [u8],
	pub up_scales: &'a [f32],
	pub down_packed: &'a [u8],
	pub down_scales: &'a [f32],
}

impl<'a> FfnBlock<'a> {
	/// Construct a borrow block from standard F32 raw memory fields.
	pub fn from_refs(gate_w: &'a [f32], up_w: &'a [f32], down_w: &'a [f32]) -> Self {
		Self {
			gate_w,
			up_w,
			down_w,
			gate_packed: &[],
			gate_scales: &[],
			up_packed: &[],
			up_scales: &[],
			down_packed: &[],
			down_scales: &[],
		}
	}

	/// Construct a borrow block from packed 4-bit quantized data targets.
	#[allow(clippy::too_many_arguments)]
	pub fn from_quantized_refs(
		gate_packed: &'a [u8],
		gate_scales: &'a [f32],
		up_packed: &'a [u8],
		up_scales: &'a [f32],
		down_packed: &'a [u8],
		down_scales: &'a [f32],
	) -> Self {
		Self {
			gate_w: &[],
			up_w: &[],
			down_w: &[],
			gate_packed,
			gate_scales,
			up_packed,
			up_scales,
			down_packed,
			down_scales,
		}
	}

	/// Dispatch token vectors into SwiGLU activation channels dynamically.
	pub fn forward(&self, input: &[f32], state: &mut FfnState, config: &ModelConfig) -> Vec<f32> {
		let n_embd = config.n_embd;
		let n_ff = config.n_ff;
		let mut out = vec![0.0f32; n_embd];

		// Evaluate if model blocks were registered with 4-bit quantized states
		let is_quantized = !self.gate_packed.is_empty();

		if is_quantized {
			// Fall back to target dequantization execution kernel (Assumes group size 32 default)
			let group_size = 32;
			swiglu_4bit_into(
				&mut out,
				input,
				self.gate_scales,
				self.gate_packed,
				self.up_scales,
				self.up_packed,
				self.down_scales,
				self.down_packed,
				n_embd,
				n_ff,
				group_size,
				&mut state.scratch_gate,
				&mut state.scratch_up,
				&mut state.scratch_act,
			);
		} else {
			// Standard linear attention pathway execution turn
			swiglu_into(
				&mut out,
				input,
				self.gate_w,
				self.up_w,
				self.down_w,
				n_embd,
				n_ff,
				&mut state.scratch_gate,
				&mut state.scratch_up,
				&mut state.scratch_act,
			);
		}

		out
	}
}

// ---------------------------------------------------------------------------
// Underlying Procedural Core Functions
// ---------------------------------------------------------------------------

pub fn swiglu(
	input: &[f32],
	gate_w: &[f32],
	up_w: &[f32],
	down_w: &[f32],
	n_embd: usize,
	n_ff: usize,
) -> Vec<f32> {
	let gate = math::gemv(gate_w, input, n_ff, n_embd);
	let up = math::gemv(up_w, input, n_ff, n_embd);
	let mut act = vec![0.0f32; n_ff];
	for i in 0..n_ff {
		act[i] = math::silu(gate[i]) * up[i];
	}
	math::gemv(down_w, &act, n_embd, n_ff)
}

pub fn swiglu_into(
	out: &mut [f32],
	input: &[f32],
	gate_w: &[f32],
	up_w: &[f32],
	down_w: &[f32],
	n_embd: usize,
	n_ff: usize,
	scratch_gate: &mut [f32],
	scratch_up: &mut [f32],
	scratch_act: &mut [f32],
) {
	math::gemv_into(scratch_gate, gate_w, input, n_ff, n_embd);
	math::gemv_into(scratch_up, up_w, input, n_ff, n_embd);
	for i in 0..n_ff {
		scratch_act[i] = math::silu(scratch_gate[i]) * scratch_up[i];
	}
	math::gemv_into(out, down_w, scratch_act, n_embd, n_ff);
}

pub fn swiglu_4bit_into(
	out: &mut [f32],
	input: &[f32],
	gate_scales: &[f32],
	gate_packed: &[u8],
	up_scales: &[f32],
	up_packed: &[u8],
	down_scales: &[f32],
	down_packed: &[u8],
	n_embd: usize,
	n_ff: usize,
	group_size: usize,
	scratch_gate: &mut [f32],
	scratch_up: &mut [f32],
	scratch_act: &mut [f32],
) {
	math::gemv_4bit_into(
		scratch_gate,
		gate_scales,
		gate_packed,
		input,
		n_ff,
		n_embd,
		group_size,
	);
	math::gemv_4bit_into(
		scratch_up, up_scales, up_packed, input, n_ff, n_embd, group_size,
	);
	for i in 0..n_ff {
		scratch_act[i] = math::silu(scratch_gate[i]) * scratch_up[i];
	}
	math::gemv_4bit_into(
		out,
		down_scales,
		down_packed,
		scratch_act,
		n_embd,
		n_ff,
		group_size,
	);
}

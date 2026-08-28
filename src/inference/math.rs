use std::arch::x86_64::*;

// ============================================================================
// ── SECTION 1: CORE MATH PRIMITIVES & OPERATORS ─────────────────────────────
// ============================================================================

pub fn gemv(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
	let mut out = vec![0.0f32; rows];
	gemv_into(&mut out, w, x, rows, cols);
	out
}

fn gemv_serial_into(out: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
	for i in 0..rows {
		let row = &w[i * cols..(i + 1) * cols];
		let mut sum = 0.0f32;
		for j in 0..cols {
			sum += row[j] * x[j];
		}
		out[i] = sum;
	}
}

pub fn gemv_into(out: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
	if w.is_empty() || x.is_empty() || rows == 0 || cols == 0 {
		for v in out.iter_mut() {
			*v = 0.0;
		}
		return;
	}
	let total_work = rows * cols;
	if total_work <= 1_000_000 {
		gemv_serial_into(out, w, x, rows, cols);
		return;
	}
	let nthreads = std::thread::available_parallelism()
		.map(|n| n.get())
		.unwrap_or(4)
		.min(32)
		.max(1);
	let chunk = (rows + nthreads - 1) / nthreads;

	std::thread::scope(|s| {
		let mut out_slice = out;
		let mut start = 0usize;
		for t in 0..nthreads {
			let end = ((t + 1) * chunk).min(rows);
			if start >= end {
				continue;
			}
			let row_count = end - start;
			let w_slice = &w[start * cols..end * cols];
			let (left, right) = out_slice.split_at_mut(row_count);
			out_slice = right;
			s.spawn(move || gemv_serial_into(left, w_slice, x, row_count, cols));
			start = end;
		}
	});
}

pub fn rms_norm_into(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
	let n = x.len();
	let mut ss = 0.0f32;
	for &v in x {
		ss += v * v;
	}
	let inv = 1.0f32 / ((ss / n as f32) + eps).sqrt();
	for i in 0..n {
		out[i] = x[i] * inv * weight[i];
	}
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
	let mut out = vec![0.0f32; x.len()];
	rms_norm_into(&mut out, x, weight, eps);
	out
}

pub fn l2_norm(x: &mut [f32], eps: f32) {
	let mut norm = 0.0f32;
	for &v in x.iter() {
		norm += v * v;
	}
	let inv = 1.0f32 / norm.sqrt().max(eps);
	for v in x.iter_mut() {
		*v *= inv;
	}
}

#[inline]
pub fn silu(x: f32) -> f32 {
	x * sigmoid(x)
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
	if x >= 0.0 {
		1.0 / (1.0 + (-x).exp())
	} else {
		let e = x.exp();
		e / (1.0 + e)
	}
}

#[inline]
pub fn softplus(x: f32) -> f32 {
	if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

pub fn softmax(x: &mut [f32]) {
	let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
	let mut sum = 0.0f32;
	for v in x.iter_mut() {
		*v = (*v - max).exp();
		sum += *v;
	}
	let inv = 1.0f32 / sum;
	for v in x.iter_mut() {
		*v *= inv;
	}
}

pub fn rope_multi(x: &mut [f32], pos: usize, _n_rot: usize, sections: [i32; 4], freq_base: f32) {
	let mut total_d = 0usize;
	for &section in &sections {
		if section > 0 {
			total_d += section as usize;
		}
	}
	if total_d == 0 {
		return;
	}

	let mut offset = 0usize;
	let mut freq_idx = 0usize;

	for &section in &sections {
		let d = if section <= 0 { 0 } else { section as usize };
		if d == 0 {
			continue;
		}

		for _ in 0..d {
			let dim_idx = offset + 2 * (freq_idx - offset / 2);
			if dim_idx + 1 >= x.len() {
				break;
			}

			let theta = pos as f32 * freq_base.powf(-(freq_idx as f32) / (total_d as f32));
			let (sin, cos) = theta.sin_cos();

			let x0 = x[dim_idx];
			let x1 = x[dim_idx + 1];
			x[dim_idx] = x0 * cos - x1 * sin;
			x[dim_idx + 1] = x0 * sin + x1 * cos;

			freq_idx += 1;
		}
		offset += 2 * d;
	}
}

pub fn conv1d_depthwise(
	input: &[f32],
	kernel: &[f32],
	kernel_size: usize,
	channels: usize,
	state: &mut Vec<f32>,
) -> Vec<f32> {
	let mut output = vec![0.0f32; channels];
	conv1d_depthwise_into(&mut output, input, kernel, kernel_size, channels, state);
	output
}

pub fn conv1d_depthwise_into(
	out: &mut [f32],
	input: &[f32],
	kernel: &[f32],
	kernel_size: usize,
	channels: usize,
	state: &mut Vec<f32>,
) {
	if kernel_size < 1 {
		return;
	}

	for c in 0..channels {
		let mut sum = 0.0f32;
		for k in 0..kernel_size - 1 {
			sum += state[k * channels + c] * kernel[c * kernel_size + k];
		}
		sum += input[c] * kernel[c * kernel_size + (kernel_size - 1)];
		out[c] = sum;
	}

	let history_steps = kernel_size - 1;
	if history_steps > 1 {
		state.copy_within(channels..history_steps * channels, 0);
	}

	if history_steps > 0 {
		let start_idx = (history_steps - 1) * channels;
		state[start_idx..history_steps * channels].copy_from_slice(input);
	}
}

pub fn matvec(s: &[f32], k: &[f32], dim: usize) -> Vec<f32> {
	gemv(s, k, dim, dim)
}

/// Unquantized f32 matvec path.
pub fn matvec_into(out: &mut [f32], s: &[f32], k: &[f32], dim: usize) {
	gemv_into(out, s, k, dim, dim);
}

/// Dedicated Quantized 4-bit matvec wrapper.
pub fn matvec_4bit_into(
	out: &mut [f32],
	scales: &[f32],
	packed: &[u8],
	x: &[f32],
	dim: usize,
	group_size: usize,
) {
	gemv_4bit_into(out, scales, packed, x, dim, dim, group_size);
}

// ============================================================================
// ── SECTION 2: HARDWARE-FUSED INT4 PERFORMANCE ACCELERATION ─────────────────
// ============================================================================

pub const GROUP_SIZE: usize = 32;

pub fn quantize(weights: &[f32], group_size: usize) -> (Vec<f32>, Vec<u8>) {
	let gs = group_size.max(1);
	let n_groups = (weights.len() + gs - 1) / gs;
	let mut scales = Vec::with_capacity(n_groups);
	let mut packed = vec![0u8; (weights.len() + 1) / 2];

	for g in 0..n_groups {
		let start = g * gs;
		let end = (start + gs).min(weights.len());
		let group = &weights[start..end];

		let max_abs = group.iter().fold(0.0f32, |a, &w| a.max(w.abs()));
		let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 0.0 };
		let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

		scales.push(scale);

		for (i, &w) in group.iter().enumerate() {
			let q = if scale > 0.0 {
				(w * inv_scale).round() as i32
			} else {
				0
			};
			let clamped = q.clamp(-8, 7) + 8;
			let idx = start + i;

			if idx % 2 == 0 {
				packed[idx / 2] |= clamped as u8;
			} else {
				packed[idx / 2] |= (clamped as u8) << 4;
			}
		}
	}
	(scales, packed)
}

pub fn gemv_4bit_into(
	out: &mut [f32],
	scales: &[f32],
	packed: &[u8],
	x: &[f32],
	rows: usize,
	cols: usize,
	group_size: usize,
) {
	if rows == 0 || cols == 0 || x.is_empty() {
		for v in out.iter_mut() {
			*v = 0.0;
		}
		return;
	}

	let total_work = rows * cols;
	if total_work <= 1_000_000 {
		gemv_4bit_worker_dispatch(out, scales, packed, x, 0, rows, cols, group_size);
		return;
	}

	let nthreads = std::thread::available_parallelism()
		.map(|n| n.get())
		.unwrap_or(4)
		.min(32)
		.max(1);
	let chunk = (rows + nthreads - 1) / nthreads;

	std::thread::scope(|s| {
		let mut out_slice = out;
		let mut start = 0usize;
		for t in 0..nthreads {
			let end = ((t + 1) * chunk).min(rows);
			if start >= end {
				continue;
			}
			let row_count = end - start;

			let (left, right) = out_slice.split_at_mut(row_count);
			out_slice = right;

			s.spawn(move || {
				gemv_4bit_worker_dispatch(left, scales, packed, x, start, end, cols, group_size);
			});
			start = end;
		}
	});
}

fn gemv_4bit_worker_dispatch(
	out: &mut [f32],
	global_scales: &[f32],
	global_packed: &[u8],
	x: &[f32],
	row_start: usize,
	row_end: usize,
	cols: usize,
	group_size: usize,
) {
	if is_x86_feature_detected!("avx512f")
		&& is_x86_feature_detected!("avx512bw")
		&& group_size == 32
		&& cols % 128 == 0
	{
		unsafe {
			gemv_4bit_parallel_worker_avx512(
				out,
				global_scales,
				global_packed,
				x,
				row_start,
				row_end,
				cols,
			);
		}
	} else {
		gemv_4bit_parallel_worker_scalar(
			out,
			global_scales,
			global_packed,
			x,
			row_start,
			row_end,
			cols,
			group_size,
		);
	}
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn gemv_4bit_parallel_worker_avx512(
	out: &mut [f32],
	global_scales: &[f32],
	global_packed: &[u8],
	x: &[f32],
	row_start: usize,
	row_end: usize,
	cols: usize,
) {
	let n_gpr = cols / 32;
	let chunks_128 = cols / 128;

	let v_offset = _mm512_set1_ps(-8.0);
	let v_mask_low = _mm512_set1_epi8(0x0F);

	for i in row_start..row_end {
		let row_scales = &global_scales[i * n_gpr..(i + 1) * n_gpr];
		let row_offset_bytes = (i * cols) / 2;
		let mut v_acc = _mm512_setzero_ps();

		for c in 0..chunks_128 {
			let byte_offset = row_offset_bytes + (c * 64);
			let x_offset = c * 128;
			let group_offset = c * 4;

			let raw_bytes = _mm512_loadu_si512(global_packed.as_ptr().add(byte_offset) as *const _);
			let bytes_even = _mm512_and_si512(raw_bytes, v_mask_low);
			let bytes_odd = _mm512_and_si512(_mm512_srli_epi16(raw_bytes, 4), v_mask_low);

			let lin_0_63 = _mm512_unpacklo_epi8(bytes_even, bytes_odd);

			let lin_64_127 = _mm512_unpackhi_epi8(bytes_even, bytes_odd);

			// --- Block 0 ---
			let scale_0 = _mm512_set1_ps(row_scales[group_offset]);
			let w0_1 = _mm512_fmadd_ps(
				_mm512_add_ps(
					_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(lin_0_63))),
					v_offset,
				),
				scale_0,
				_mm512_setzero_ps(),
			);
			let w0_2 = _mm512_fmadd_ps(
				_mm512_add_ps(
					_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(
						lin_0_63, 1,
					))),
					v_offset,
				),
				scale_0,
				_mm512_setzero_ps(),
			);
			v_acc = _mm512_fmadd_ps(w0_1, _mm512_loadu_ps(x.as_ptr().add(x_offset)), v_acc);
			v_acc = _mm512_fmadd_ps(w0_2, _mm512_loadu_ps(x.as_ptr().add(x_offset + 16)), v_acc);

			// --- Block 1 ---
			let scale_1 = _mm512_set1_ps(row_scales[group_offset + 1]);
			let w1_1 = _mm512_fmadd_ps(
				_mm512_add_ps(
					_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(
						lin_0_63, 2,
					))),
					v_offset,
				),
				scale_1,
				_mm512_setzero_ps(),
			);
			let w1_2 = _mm512_fmadd_ps(
				_mm512_add_ps(
					_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(
						lin_0_63, 3,
					))),
					v_offset,
				),
				scale_1,
				_mm512_setzero_ps(),
			);
			v_acc = _mm512_fmadd_ps(w1_1, _mm512_loadu_ps(x.as_ptr().add(x_offset + 32)), v_acc);
			v_acc = _mm512_fmadd_ps(w1_2, _mm512_loadu_ps(x.as_ptr().add(x_offset + 48)), v_acc);

			// --- Block 2 ---
			let scale_2 = _mm512_set1_ps(row_scales[group_offset + 2]);
			let w2_1 = _mm512_fmadd_ps(
				_mm512_add_ps(
					_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(lin_64_127))),
					v_offset,
				),
				scale_2,
				_mm512_setzero_ps(),
			);
			let w2_2 = _mm512_fmadd_ps(
				_mm512_add_ps(
					_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(
						lin_64_127, 1,
					))),
					v_offset,
				),
				scale_2,
				_mm512_setzero_ps(),
			);
			v_acc = _mm512_fmadd_ps(w2_1, _mm512_loadu_ps(x.as_ptr().add(x_offset + 64)), v_acc);
			v_acc = _mm512_fmadd_ps(w2_2, _mm512_loadu_ps(x.as_ptr().add(x_offset + 80)), v_acc);

			// --- Block 3 ---
			let scale_3 = _mm512_set1_ps(row_scales[group_offset + 3]);
			let w3_1 = _mm512_fmadd_ps(
				_mm512_add_ps(
					_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(
						lin_64_127, 2,
					))),
					v_offset,
				),
				scale_3,
				_mm512_setzero_ps(),
			);
			let w3_2 = _mm512_fmadd_ps(
				_mm512_add_ps(
					_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(
						lin_64_127, 3,
					))),
					v_offset,
				),
				scale_3,
				_mm512_setzero_ps(),
			);
			v_acc = _mm512_fmadd_ps(w3_1, _mm512_loadu_ps(x.as_ptr().add(x_offset + 96)), v_acc);
			v_acc = _mm512_fmadd_ps(w3_2, _mm512_loadu_ps(x.as_ptr().add(x_offset + 112)), v_acc);
		}
		out[i - row_start] = _mm512_reduce_add_ps(v_acc);
	}
}

pub fn gemv_4bit_parallel_worker_scalar(
	out: &mut [f32],
	global_scales: &[f32],
	global_packed: &[u8],
	x: &[f32],
	row_start: usize,
	row_end: usize,
	cols: usize,
	group_size: usize,
) {
	let gs = group_size.max(1);
	let n_gpr = (cols + gs - 1) / gs;

	for i in row_start..row_end {
		let row_scales = &global_scales[i * n_gpr..(i + 1) * n_gpr];
		let row_offset = i * cols;
		let mut sum = 0.0f32;

		for g in 0..n_gpr {
			let scale = row_scales[g];
			let j_start = g * gs;
			let j_end = (j_start + gs).min(cols);

			for j in j_start..j_end {
				let linear = row_offset + j;
				let nibble = if linear & 1 == 0 {
					global_packed[linear >> 1] & 0x0F
				} else {
					(global_packed[linear >> 1] >> 4) & 0x0F
				};
				let w = (nibble as f32 - 8.0) * scale;
				sum += w * x[j];
			}
		}
		out[i - row_start] = sum;
	}
}

// ============================================================================
// ── SECTION 3: UNIT TESTS ───────────────────────────────────────────────────
// ============================================================================

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_rope_multi() {
		let mut x = vec![1.0, 0.0, 0.0, 1.0];
		rope_multi(&mut x, 1, 4, [1, 1, 0, 0], 10000.0);
		assert!(x[0] != 1.0 || x[1] != 0.0);
	}

	#[test]
	fn test_conv1d() {
		// Correct layout arrangement for 3 channels with a kernel size of 2
		// Each channel row owns exactly 2 elements: [history_weight, input_weight]
		let kernel = vec![
			1.0, 1.0, // Channel 0 weights
			2.0, 1.0, // Channel 1 weights
			3.0, 1.0, // Channel 2 weights
		];
		let input = vec![1.0, 2.0, 3.0];
		let mut state = vec![0.0; 3];

		let out = conv1d_depthwise(&input, &kernel, 2, 3, &mut state);

		assert_eq!(out, vec![1.0, 2.0, 3.0]);
		assert_eq!(state, vec![1.0, 2.0, 3.0]);
	}

	#[test]
	fn test_gemv_4bit_execution() {
		// Test basic end-to-end functionality of your 4-bit GEMV engine
		let rows = 4;
		let cols = 128; // Keep aligned to 128 for AVX-512 target branches
		let weights = vec![0.5f32; rows * cols];
		let x = vec![1.0f32; cols];

		let (scales, packed) = quantize(&weights, 32);
		let mut out = vec![0.0f32; rows];

		gemv_4bit_into(&mut out, &scales, &packed, &x, rows, cols, 32);

		// Assert execution results aren't zeroed out completely
		assert!(out[0] > 0.0);
	}
}

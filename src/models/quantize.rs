//! Quantization: CPU (AVX512) and GPU (shader).
//!
//! Both functions take a model in any format (not sandbag — sandbag is
//! already quantized). Weights are quantized sequentially so every tensor
//! lands at its original position in the output file. No decompression
//! pipeline. No rearranging.

use crate::models::format::*;
use crate::models::tensor::*;
use std::io::Write;

// ---------------------------------------------------------------------------
// CPU: AVX512 quantize
// ---------------------------------------------------------------------------
/// Quantize a model on CPU using AVX512 intrinsics.
///
/// Reads tensors from `raw` at their src_offset, quantizes each block,
/// and writes the sandbag file to `dst_path`.
pub fn quantize_cpu(
	tensors: &[Tensor],
	raw: &[u8],
	dst_path: &std::path::Path,
) -> Result<(), String> {
	// MaxAbs, not a clipped threshold. Percentile/KL calibration is built for
	// activations, whose long tails come from rare outlier tokens worth clipping.
	// Weight distributions are compact, every large weight carries signal, and a
	// 32-element block scale already adapts locally — so clipping only loses data.
	// Measured on Qwen3.5-0.8B: MaxAbs 0.0037% rel RMS vs Percentile(99.9) 10.5%.
	quantize_cpu_with(tensors, raw, dst_path, ScaleMethod::MaxAbs)
}

/// As `quantize_cpu`, with an explicit saturation calibration method.
pub fn quantize_cpu_with(
	tensors: &[Tensor],
	raw: &[u8],
	dst_path: &std::path::Path,
	method: ScaleMethod,
) -> Result<(), String> {
	// Calculate total data size first
	let mut data_size: u64 = 0;
	for t in tensors {
		data_size += sandbag_tensor_bytes(t.elem_count as u64);
	}

	// Header
	let header = SandbagHeader::new(tensors.len() as u64, data_size);
	let header_bytes = {
		let mut buf = Vec::with_capacity(24);
		buf.extend_from_slice(&header.magic.to_le_bytes());
		buf.extend_from_slice(&header.version.to_le_bytes());
		buf.extend_from_slice(&header.num_tensors.to_le_bytes());
		buf.extend_from_slice(&header.total_data_bytes.to_le_bytes());
		buf
	};

	// Index entries
	let mut index_bytes = Vec::new();
	let mut entry_offsets = Vec::with_capacity(tensors.len());
	let mut current_offset: u64 = 0;

	for t in tensors {
		let _name_bytes = t.name.as_bytes();
		let this_data_size = sandbag_tensor_bytes(t.elem_count as u64);

		let entry = SandbagTensorEntry::new(
			t.name.clone(),
			t.shape.iter().map(|&d| d as u64).collect(),
			GgmlType::Sandbag,
			current_offset,
			this_data_size,
			0, // prefix will be set during quantization
			Vec::new(),
		);

		let entry_serialized = bincode::serialize(&entry).unwrap();
		entry_offsets.push((
			entry_serialized.len() as u64,
			current_offset,
			this_data_size,
		));
		current_offset += this_data_size;
	}

	// Serialize index entries into a contiguous block
	for t in tensors {
		let name_bytes = t.name.as_bytes();
		let mut entry_buf = Vec::new();
		entry_buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
		entry_buf.extend_from_slice(name_bytes);
		let shape_dims = t.shape.len() as u64;
		entry_buf.extend_from_slice(&shape_dims.to_le_bytes());
		for &dim in &t.shape {
			entry_buf.extend_from_slice(&(dim as u64).to_le_bytes());
		}
		entry_buf.push(GgmlType::Sandbag as u8);
		index_bytes.extend_from_slice(&entry_buf);
	}

	// Data section — quantize tensors in parallel, then concatenate in the
	// original order. Order is the format's load-bearing invariant, so the
	// parallelism is in the encoding only, never in the placement.
	use rayon::prelude::*;
	let encoded: Vec<Result<Vec<u8>, String>> = tensors
		.par_iter()
		.map(|t| {
			let start = t.src_offset as usize;
			let end = start + t.src_size as usize;
			let slice = raw
				.get(start..end)
				.ok_or_else(|| format!("Tensor {} runs past end of source", t.name))?;
			let vals = decode_to_f32(slice, t.elem_count, t.dtype)
				.map_err(|e| format!("Tensor {}: {}", t.name, e))?;
			let threshold = calibrate(&vals, method);
			Ok(encode_sandbag(&vals, DEFAULT_TAIL_DIGITS, threshold))
		})
		.collect();

	let mut data_buf = Vec::with_capacity(data_size as usize);
	for chunk in encoded {
		data_buf.extend_from_slice(&chunk?);
	}

	// Write output
	let mut out = std::fs::File::create(dst_path).map_err(|e| format!("Create output: {}", e))?;
	out.write_all(&header_bytes)
		.map_err(|e| format!("Write header: {}", e))?;
	out.write_all(&index_bytes)
		.map_err(|e| format!("Write index: {}", e))?;
	out.write_all(&data_buf)
		.map_err(|e| format!("Write data: {}", e))?;

	log::info!(
		"Wrote sandbag file: {} tensors, {} data bytes",
		tensors.len(),
		data_size
	);
	Ok(())
}

/// Decimal digits kept after the 2-digit prefix. 0..=3.
///
/// A 5-digit magnitude needs 17 bits, so it cannot fit the 2-byte-per-weight
/// layout losslessly (100 × 1000 = 100,000 > 65,536). At `tail_digits` <= 2 the
/// tail is stored exactly; at 3 it is uniformly rescaled into a byte and the
/// third digit carries up to half a step of error.
pub const DEFAULT_TAIL_DIGITS: u32 = 3;

/// Widen a source block to f32, honouring the real element width.
///
/// The previous version computed a per-dtype stride and then loaded every input
/// as f32 regardless, so anything narrower than F32 was misread.
pub fn decode_to_f32(block: &[u8], elem_count: usize, dtype: GgmlType) -> Result<Vec<f32>, String> {
	let need = dtype.tensor_bytes(elem_count as u64) as usize;
	if block.len() < need {
		return Err(format!(
			"Block has {} bytes, need {} for {} x {}",
			block.len(),
			need,
			elem_count,
			dtype.name()
		));
	}

	let mut out = Vec::with_capacity(elem_count);
	match dtype {
		GgmlType::F32 => {
			for c in block[..elem_count * 4].chunks_exact(4) {
				out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
			}
		}
		GgmlType::F16 => {
			for c in block[..elem_count * 2].chunks_exact(2) {
				out.push(half::f16::from_le_bytes([c[0], c[1]]).to_f32());
			}
		}
		GgmlType::BF16 => {
			// bf16 is the top 16 bits of an f32 — widen by shifting back up.
			for c in block[..elem_count * 2].chunks_exact(2) {
				let bits = u16::from_le_bytes([c[0], c[1]]) as u32;
				out.push(f32::from_bits(bits << 16));
			}
		}
		GgmlType::F64 => {
			for c in block[..elem_count * 8].chunks_exact(8) {
				out.push(f64::from_le_bytes(c.try_into().unwrap()) as f32);
			}
		}
		other => {
			return Err(format!(
				"{} is block-quantized; dequantize before sandbag encoding",
				other.name()
			));
		}
	}
	Ok(out)
}

// ---------------------------------------------------------------------------
// Saturation calibration
// ---------------------------------------------------------------------------
/// How to pick the per-tensor saturation threshold that block scales clamp to.
///
/// Max-abs lets a single outlier set the scale for its whole block, which costs
/// resolution for every other weight in it. Percentile and KL both trade a
/// little clipping for a lot of resolution on the bulk of the distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScaleMethod {
	/// scale = max |w| in each block. No clipping, worst resolution.
	MaxAbs,
	/// Clamp at the given percentile of |w| over the tensor, e.g. 99.9.
	Percentile(f32),
	/// TensorRT-style: choose the threshold whose quantized distribution has the
	/// lowest KL divergence from the original.
	KlDivergence,
}

/// Bins used for calibration histograms.
const CALIB_BINS: usize = 2048;
/// Quantization levels the KL search models — the tail resolution of one block.
const CALIB_LEVELS: usize = 256;

/// Histogram of |w| over [0, max], plus that max.
fn abs_histogram(vals: &[f32]) -> (Vec<f64>, f32) {
	let mut max = 0.0f32;
	for &v in vals {
		let a = v.abs();
		if a > max {
			max = a;
		}
	}
	if max <= 0.0 {
		return (vec![0.0; CALIB_BINS], 0.0);
	}
	let mut hist = vec![0.0f64; CALIB_BINS];
	let scale = CALIB_BINS as f32 / max;
	for &v in vals {
		let mut b = (v.abs() * scale) as usize;
		if b >= CALIB_BINS {
			b = CALIB_BINS - 1;
		}
		hist[b] += 1.0;
	}
	(hist, max)
}

/// Threshold at the p-th percentile of |w|, read off the histogram.
fn percentile_threshold(hist: &[f64], max: f32, p: f32) -> f32 {
	let total: f64 = hist.iter().sum();
	if total <= 0.0 {
		return max;
	}
	let target = total * (p as f64 / 100.0);
	let mut acc = 0.0;
	for (i, &c) in hist.iter().enumerate() {
		acc += c;
		if acc >= target {
			return (i + 1) as f32 * max / CALIB_BINS as f32;
		}
	}
	max
}

/// KL-divergence-optimal saturation threshold (TensorRT calibration).
///
/// For each candidate cutoff i, everything at or beyond i is collapsed into the
/// last kept bin (that is what saturation does), the result is requantized to
/// CALIB_LEVELS and expanded back to i bins, and the divergence between the two
/// distributions is measured. The cutoff with the least divergence wins.
fn kl_threshold(hist: &[f64], max: f32) -> f32 {
	if max <= 0.0 {
		return 0.0;
	}
	let mut best_i = CALIB_BINS;
	let mut best_kl = f64::INFINITY;

	for i in CALIB_LEVELS..=CALIB_BINS {
		// P: the reference, with the tail folded into the boundary bin.
		let mut p: Vec<f64> = hist[..i].to_vec();
		let outliers: f64 = hist[i..].iter().sum();
		p[i - 1] += outliers;

		let p_sum: f64 = p.iter().sum();
		if p_sum <= 0.0 {
			continue;
		}

		// Q: P collapsed to CALIB_LEVELS and stretched back out, spreading each
		// level's mass only over the bins that actually held samples.
		let mut q = vec![0.0f64; i];
		for l in 0..CALIB_LEVELS {
			let lo = l * i / CALIB_LEVELS;
			let hi = ((l + 1) * i / CALIB_LEVELS).min(i);
			if hi <= lo {
				continue;
			}
			let mass: f64 = hist[lo..hi].iter().sum();
			let occupied = hist[lo..hi].iter().filter(|&&c| c > 0.0).count();
			if occupied == 0 || mass <= 0.0 {
				continue;
			}
			let share = mass / occupied as f64;
			for b in lo..hi {
				if hist[b] > 0.0 {
					q[b] = share;
				}
			}
		}

		let q_sum: f64 = q.iter().sum();
		if q_sum <= 0.0 {
			continue;
		}

		let mut kl = 0.0f64;
		for b in 0..i {
			let pv = p[b] / p_sum;
			let qv = q[b] / q_sum;
			if pv > 0.0 && qv > 0.0 {
				kl += pv * (pv / qv).ln();
			}
		}

		if kl < best_kl {
			best_kl = kl;
			best_i = i;
		}
	}

	best_i as f32 * max / CALIB_BINS as f32
}

/// Resolve a method to a concrete saturation threshold for one tensor.
pub fn calibrate(vals: &[f32], method: ScaleMethod) -> f32 {
	match method {
		ScaleMethod::MaxAbs => {
			let mut m = 0.0f32;
			for &v in vals {
				let a = v.abs();
				if a > m {
					m = a;
				}
			}
			m
		}
		ScaleMethod::Percentile(p) => {
			let (h, max) = abs_histogram(vals);
			percentile_threshold(&h, max, p)
		}
		ScaleMethod::KlDivergence => {
			let (h, max) = abs_histogram(vals);
			kl_threshold(&h, max)
		}
	}
}

/// Quantize one tensor's worth of f32 into the sandbag layout.
///
/// Output: `[prefix_u8, tail_u8] * N` then the sign plane as u64 words, matching
/// `format::sandbag_tensor_bytes`. Every element occupies the same two bytes, so
/// a reader can seek to element i at `2*i` with no table.
pub fn encode_sandbag(vals: &[f32], tail_digits: u32, threshold: f32) -> Vec<u8> {
	let n = vals.len();
	let block = SANDBAG_BLOCK_ELEMS as usize;
	let nblocks = n.div_ceil(block);

	let mut scales: Vec<u8> = Vec::with_capacity(nblocks * 2);
	let mut pairs: Vec<u8> = Vec::with_capacity(n * 2);
	let mut sign_words: Vec<u64> = Vec::with_capacity(n.div_ceil(64));

	let mut word: u64 = 0;

	for b in 0..nblocks {
		let lo = b * block;
		let hi = (lo + block).min(n);

		// Block scale is the largest magnitude present, capped at the calibrated
		// saturation point so one outlier cannot flatten the rest of the block.
		let mut s = 0.0f32;
		for &v in &vals[lo..hi] {
			let a = v.abs().min(threshold);
			if a > s {
				s = a;
			}
		}
		// A zero block still needs a usable divisor.
		let s_stored = half::f16::from_f32(if s > 0.0 { s } else { 1.0 });
		scales.extend_from_slice(&s_stored.to_le_bytes());
		// Divide rather than multiply by a reciprocal, so this matches the shader
		// operation for operation. Vulkan allows 2.5 ULP on division, so a
		// reciprocal-then-multiply here drifts from the GPU by 1 in the last tail
		// digit on a small fraction of weights.
		let s_div = s_stored.to_f32();

		for (k, &v) in vals[lo..hi].iter().enumerate() {
			let i = lo + k;
			if v.is_sign_negative() {
				word |= 1u64 << (i % 64);
			}
			if i % 64 == 63 {
				sign_words.push(word);
				word = 0;
			}

			// Normalized into [0,1], so all five digits are significant no matter
			// how small the weight is in absolute terms.
			let norm = (v.abs().min(threshold) / s_div).min(1.0);
			let (prefix, tail) = split_prefix_tail(norm, tail_digits);
			pairs.push(prefix);
			pairs.push(tail);
		}
	}
	if n % 64 != 0 {
		sign_words.push(word);
	}

	let mut out = Vec::with_capacity(sandbag_tensor_bytes(n as u64) as usize);
	out.extend_from_slice(&scales);
	out.extend_from_slice(&pairs);
	for w in &sign_words {
		out.extend_from_slice(&w.to_le_bytes());
	}
	out
}

/// Decode one weight back. Mirrors `encode_sandbag` for tests and error reports.
pub fn decode_sandbag_weight(
	prefix: u8,
	tail: u8,
	negative: bool,
	scale: f32,
	tail_digits: u32,
) -> f32 {
	join_prefix_tail(prefix, tail, negative, tail_digits) * scale
}

/// Legacy entry point retained for the scalar fallback path.
#[allow(dead_code)]
fn avx512_quantize_block(block: &[u8], elem_count: usize, scheme: &GgmlType) -> Vec<u8> {
	let mut out = Vec::with_capacity(elem_count * 2);
	let mut sign_bits = Vec::new();

	let bytes_per_elem = match scheme {
		GgmlType::F32 => 4,
		GgmlType::F16 | GgmlType::BF16 | GgmlType::F64 => 2,
		_ => 4,
	};

	let mut i = 0usize;
	let mut sign_word: u64 = 0;
	let mut sign_bit_pos: u32 = 0;

	#[cfg(target_arch = "x86_64")]
	{
		use std::arch::x86_64::*;

		while i + 8 <= elem_count {
			let ptr = unsafe { block.as_ptr().add(i * bytes_per_elem) };
			let raw = unsafe { _mm256_loadu_ps(ptr as *const f32) };

			// Sign check: which lanes are negative?
			let zero = unsafe { _mm256_setzero_ps() };
			let lt_zero = unsafe { _mm256_cmp_ps(raw, zero, _MM_CMPINT_LT) };
			let sign_mask_i32 = unsafe { _mm256_castps_si256(lt_zero) };

			// Absolute values
			let abs_val = unsafe { _mm256_andnot_ps(_mm256_set1_ps(-0.0), raw) };

			// Store to arrays to avoid unsafe as_core() calls
			let mut vals = [0.0f32; 8];
			let mut signs = [0i32; 8];
			unsafe {
				_mm256_storeu_ps(vals.as_mut_ptr(), abs_val);
				_mm256_storeu_si256(signs.as_mut_ptr() as *mut _, sign_mask_i32);
			}

			for j in 0..8 {
				let val = vals[j];
				let negative = signs[j] != 0;

				if negative {
					sign_word |= 1 << sign_bit_pos;
				}
				sign_bit_pos += 1;

				if sign_bit_pos == 64 {
					sign_bits.push(sign_word);
					sign_word = 0;
					sign_bit_pos = 0;
				}

				// Quantize: val → prefix (2-digit) + tail (3-digit)
				let (prefix, tail) = quantize_f32_to_prefix_tail(val);
				out.push(prefix);
				out.push(tail);
			}

			i += 8;
		}
	}

	#[cfg(not(target_arch = "x86_64"))]
	{
		// Fallback scalar path
		while i < elem_count {
			let offset = i * bytes_per_elem;
			let val = if bytes_per_elem == 4 {
				f32::from_le_bytes(block[offset..offset + 4].try_into().unwrap())
			} else if bytes_per_elem == 2 {
				f16::from_le_bytes(block[offset..offset + 2].try_into().unwrap()).to_f32()
			} else {
				0.0
			};

			let negative = val.is_sign_negative();
			let abs_val = val.abs();

			if negative {
				sign_word |= 1 << sign_bit_pos;
			}
			sign_bit_pos += 1;
			if sign_bit_pos == 64 {
				sign_bits.push(sign_word);
				sign_word = 0;
				sign_bit_pos = 0;
			}

			let (prefix, tail) = quantize_f32_to_prefix_tail(abs_val);
			out.push(prefix);
			out.push(tail);

			i += 1;
		}
	}

	// Pad remaining sign bits
	if sign_bit_pos > 0 {
		sign_bits.push(sign_word);
	}

	// Append sign bits after the prefix/tail bytes
	for word in &sign_bits {
		out.extend_from_slice(&word.to_le_bytes());
	}

	out
}

/// Split a non-negative magnitude into (prefix, tail) decimal digits.
///
/// `0.2568939` with 3 tail digits gives prefix `25`, tail `689` — the first two
/// digits and the next three, concatenated rather than summed on decode.
///
/// The old version took `scaled % 10_000` (the low **four** digits) and then
/// clamped to 999, which pinned almost every tail to exactly 999. Taking the
/// next three digits means dividing that remainder by ten.
pub fn split_prefix_tail(val: f32, tail_digits: u32) -> (u8, u8) {
	// Five significant decimal digits of the magnitude: 0..=99_999.
	let scaled = (val * 100_000.0).round().clamp(0.0, 99_999.0) as u32;

	let prefix = (scaled / 1_000).min(99) as u8;
	let rest = scaled % 1_000; // the three digits after the prefix

	let tail = match tail_digits {
		0 => 0u8,
		1 => (rest / 100) as u8, // 0..=9    exact
		2 => (rest / 10) as u8,  // 0..=99   exact
		// 0..=999 cannot fit a byte; rescale uniformly and note the loss.
		_ => ((rest as u32 * 255 + 499) / 999) as u8,
	};
	(prefix, tail)
}

/// Inverse of `split_prefix_tail` — used by tests and by the error report in the
/// bench. Inference decodes inline in the GEMV kernel rather than calling this.
pub fn join_prefix_tail(prefix: u8, tail: u8, negative: bool, tail_digits: u32) -> f32 {
	let rest = match tail_digits {
		0 => 0u32,
		1 => tail as u32 * 100,
		2 => tail as u32 * 10,
		_ => (tail as u32 * 999 + 127) / 255,
	};
	let scaled = prefix as u32 * 1_000 + rest.min(999);
	let mag = scaled as f32 / 100_000.0;
	if negative { -mag } else { mag }
}

/// Retained for the legacy scalar path.
#[allow(dead_code)]
fn quantize_f32_to_prefix_tail(val: f32) -> (u8, u8) {
	split_prefix_tail(val, DEFAULT_TAIL_DIGITS)
}

// ---------------------------------------------------------------------------
// GPU: quantize via shader submission
// ---------------------------------------------------------------------------
/// Submit a model to the GPU for quantization via the cached shader pipeline.
///
/// The shader writes the sandbag format sequentially:
///   every weight in its original position, nothing rearranged.
///
/// This function requires a initialized MemoryController (see controller.rs).
pub fn quantize_gpu(
	tensors: &[Tensor],
	raw: &[u8],
	dst_path: &std::path::Path,
) -> Result<(), String> {
	use crate::memory_controller::controller::GLOBAL_CONTROLLER;

	let ctrl = GLOBAL_CONTROLLER
		.get()
		.ok_or("MemoryController not initialized")?;
	let mut ctrl = ctrl.lock().map_err(|e| format!("Lock poison: {}", e))?;

	if ctrl.gpu.cached_quantize_pipeline == ash::vk::Pipeline::null() {
		return Err(
			"GPU quantize unavailable: sandbag_quantize.spv was not found at startup. \
			 Build it with: glslangValidator --target-env vulkan1.3 \
			 -o src/models/sandbag_quantize.spv src/models/sandbag_quantize.comp"
				.into(),
		);
	}

	// Source weights are paged in back to back, so tensor n starts at the sum of
	// the sizes of tensors 0..n. The layout below mirrors that on the output side.
	let blocks: Vec<crate::memory_controller::controller::BlockDescriptor> = tensors
		.iter()
		.map(|t| crate::memory_controller::controller::BlockDescriptor {
			offset: t.src_offset,
			size: t.src_size,
		})
		.collect();

	let src_bytes = ctrl.submit_blocks_for_paging(raw, &blocks)?;

	// Destination region begins after the source, page-aligned so the two never
	// share a page. Its pages must be committed before the shader writes to them:
	// the arena is sparse, so uncommitted addresses have no memory behind them.
	let page_size = ctrl.arena.page_size;
	let dst_base = src_bytes.next_multiple_of(page_size);
	let dst_bytes: u64 = tensors
		.iter()
		.map(|t| sandbag_tensor_bytes(t.elem_count as u64))
		.sum();

	let first_page = (dst_base / page_size) as usize;
	let last_page = ((dst_base + dst_bytes).div_ceil(page_size)) as usize;
	if last_page > ctrl.arena.total_pages {
		return Err(format!(
			"Output needs pages up to {} but the arena has {}",
			last_page, ctrl.arena.total_pages
		));
	}
	for p in first_page..last_page {
		ctrl.commit_page(p);
	}

	dispatch_quantize_shader(
		&ctrl,
		tensors,
		raw,
		ScaleMethod::MaxAbs,
		dst_base,
		dst_bytes,
		dst_path,
	)
}

/// Dispatch the quantize compute shader bound to the sparse buffer arena.
///
/// One dispatch per tensor, each carrying its own source/destination offsets in
/// push constants. Both regions live in the same flat arena binding, so the
/// descriptor set is bound once and never rewritten.
#[allow(clippy::too_many_arguments)]
fn dispatch_quantize_shader(
	ctrl: &std::sync::MutexGuard<'_, crate::memory_controller::controller::MemoryController>,
	tensors: &[Tensor],
	raw_for_calibration: &[u8],
	method: ScaleMethod,
	dst_base: u64,
	dst_bytes: u64,
	dst_path: &std::path::Path,
) -> Result<(), String> {
	use crate::memory_controller::controller::QuantizePushConstants;
	use ash::vk;

	/// Must match `GROUP` and `local_size_x` in sandbag_quantize.comp.
	const ELEMS_PER_INVOCATION: u64 = 64;
	const LOCAL_SIZE: u64 = 64;

	let device = ctrl.gpu.device();
	let queue = ctrl.gpu.queue();

	// Allocate command buffer
	let cmd = crate::memory_controller::controller::GpuContext::alloc_cmd_buffer(
		device,
		ctrl.gpu.command_pool,
		&ctrl.gpu.cmd_buffer_pool,
	);

	unsafe {
		device
			.begin_command_buffer(
				cmd,
				&vk::CommandBufferBeginInfo::default()
					.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
			)
			.expect("begin cmd buffer for quantize");

		device.cmd_bind_pipeline(
			cmd,
			vk::PipelineBindPoint::COMPUTE,
			ctrl.gpu.cached_quantize_pipeline,
		);
		device.cmd_bind_descriptor_sets(
			cmd,
			vk::PipelineBindPoint::COMPUTE,
			ctrl.gpu.cached_pipeline_layout,
			0,
			&[ctrl.gpu.cached_descriptor_set],
			&[],
		);
	}

	// Walk tensors in order, tracking both cursors. Source and destination
	// offsets are both running sums — that is the sequential invariant.
	let mut src_cursor: u64 = 0;
	let mut dst_cursor: u64 = dst_base;

	for t in tensors {
		let src_type = match t.dtype {
			GgmlType::F32 => 0u32,
			GgmlType::F16 => 1,
			GgmlType::BF16 => 2,
			other => {
				return Err(format!(
					"Tensor {}: {} is block-quantized; dequantize before sandbag encoding",
					t.name,
					other.name()
				));
			}
		};

		let n = t.elem_count as u64;

		// The shader needs a saturation threshold, and calibration needs a
		// whole-tensor histogram — a reduction the encode kernel cannot do while
		// each invocation owns an independent 64-weight slice. So it is computed
		// host-side and handed over as a scalar.
		let threshold = {
			let start = t.src_offset as usize;
			let end = start + t.src_size as usize;
			let slice = raw_for_calibration
				.get(start..end)
				.ok_or_else(|| format!("Tensor {} runs past end of source", t.name))?;
			let vals = decode_to_f32(slice, t.elem_count, t.dtype)
				.map_err(|e| format!("Tensor {}: {}", t.name, e))?;
			calibrate(&vals, method)
		};

		let off = |v: u64, what: &str| -> Result<u32, String> {
			u32::try_from(v).map_err(|_| format!("Tensor {} {} exceeds 4 GiB", t.name, what))
		};

		let pc = QuantizePushConstants {
			src_offset: off(src_cursor, "source offset")?,
			scale_offset: off(
				dst_cursor + sandbag_scale_offset_in_tensor(),
				"scale offset",
			)?,
			pairs_offset: off(dst_cursor + sandbag_pairs_offset(n), "pairs offset")?,
			sign_offset: off(dst_cursor + sandbag_sign_offset(n), "sign offset")?,
			elem_count: n as u32,
			src_type,
			tail_digits: DEFAULT_TAIL_DIGITS,
			threshold,
		};

		let bytes = unsafe {
			std::slice::from_raw_parts(
				&pc as *const _ as *const u8,
				std::mem::size_of::<QuantizePushConstants>(),
			)
		};

		// One invocation per 32 weights, LOCAL_SIZE invocations per workgroup.
		let groups = n.div_ceil(ELEMS_PER_INVOCATION).div_ceil(LOCAL_SIZE).max(1);

		unsafe {
			device.cmd_push_constants(
				cmd,
				ctrl.gpu.cached_pipeline_layout,
				vk::ShaderStageFlags::COMPUTE,
				0,
				bytes,
			);
			device.cmd_dispatch(cmd, groups as u32, 1, 1);
		}

		src_cursor += t.src_size;
		dst_cursor += sandbag_tensor_bytes(n);
	}

	// Make the writes visible to the host read that follows.
	unsafe {
		let barrier = vk::MemoryBarrier::default()
			.src_access_mask(vk::AccessFlags::SHADER_WRITE)
			.dst_access_mask(vk::AccessFlags::HOST_READ);
		device.cmd_pipeline_barrier(
			cmd,
			vk::PipelineStageFlags::COMPUTE_SHADER,
			vk::PipelineStageFlags::HOST,
			vk::DependencyFlags::empty(),
			&[barrier],
			&[],
			&[],
		);
		device
			.end_command_buffer(cmd)
			.expect("end cmd buffer for quantize");
	}

	// Submit and wait
	let fence =
		crate::memory_controller::controller::GpuContext::alloc_fence(device, &ctrl.gpu.fence_pool);

	unsafe {
		device
			.queue_submit(
				queue,
				&[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
				fence,
			)
			.expect("submit quantize");

		device
			.wait_for_fences(&[fence], true, u64::MAX)
			.expect("wait quantize fence");
	}

	// Read back the destination region only — the source half of the arena is
	// still holding the original weights.
	let result = unsafe {
		ctrl.gpu.download(
			ctrl.arena.sparse_buffer,
			dst_base as vk::DeviceSize,
			dst_bytes as vk::DeviceSize,
		)
	};

	// Recycle resources
	unsafe {
		crate::memory_controller::controller::GpuContext::recycle_fence(
			device,
			fence,
			&ctrl.gpu.fence_pool,
		);
		device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty());
		ctrl.gpu.cmd_buffer_pool.lock().unwrap().push(cmd);
	}

	// Write sandbag output
	let header = SandbagHeader::new(tensors.len() as u64, result.len() as u64);
	let mut out = std::fs::File::create(dst_path).map_err(|e| format!("Create output: {}", e))?;

	out.write_all(&header.magic.to_le_bytes())
		.map_err(|e| format!("Write: {}", e))?;
	out.write_all(&header.version.to_le_bytes())
		.map_err(|e| format!("Write: {}", e))?;
	out.write_all(&header.num_tensors.to_le_bytes())
		.map_err(|e| format!("Write: {}", e))?;
	out.write_all(&header.total_data_bytes.to_le_bytes())
		.map_err(|e| format!("Write: {}", e))?;
	out.write_all(&result)
		.map_err(|e| format!("Write data: {}", e))?;

	log::info!("GPU quantize complete: {} bytes written", result.len());
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The worked example from the format spec.
	#[test]
	fn spec_example_splits_into_25_and_689() {
		let (prefix, tail) = split_prefix_tail(0.2568939, 2);
		assert_eq!(prefix, 25);
		assert_eq!(tail, 68); // 2-digit tail keeps 68 of 689

		let (prefix, _) = split_prefix_tail(0.2568939, 3);
		assert_eq!(prefix, 25);
		// The old code returned tail 999 here for essentially every input.
		let scaled = (0.2568939f32 * 100_000.0).round() as u32;
		assert_eq!(scaled % 1_000, 689);
	}

	#[test]
	fn two_digit_tail_round_trips_exactly() {
		for i in 0..10_000u32 {
			let v = i as f32 / 100_000.0; // 0.00000 .. 0.09999
			let (p, t) = split_prefix_tail(v, 2);
			let back = join_prefix_tail(p, t, false, 2);
			// 4 significant digits retained => error under one unit of the 4th.
			assert!(
				(back - v).abs() <= 1.0 / 10_000.0 + 1e-9,
				"v={} back={} p={} t={}",
				v,
				back,
				p,
				t
			);
		}
	}

	#[test]
	fn sign_plane_is_u64_words_at_the_end() {
		let vals: Vec<f32> = (0..100)
			.map(|i| if i % 3 == 0 { -0.5 } else { 0.5 })
			.collect();
		let threshold = calibrate(&vals, ScaleMethod::MaxAbs);
		let enc = encode_sandbag(&vals, DEFAULT_TAIL_DIGITS, threshold);

		let expected = sandbag_tensor_bytes(100);
		assert_eq!(enc.len() as u64, expected, "layout must match the reader");

		let sign_start = sandbag_sign_offset(100) as usize;
		let w0 = u64::from_le_bytes(enc[sign_start..sign_start + 8].try_into().unwrap());
		for i in 0..64 {
			assert_eq!(w0 >> i & 1 == 1, i % 3 == 0, "sign bit {}", i);
		}
	}

	/// Per-block scaling is what makes small weights survive: relative error must
	/// stay flat as the block magnitude drops, which is exactly what the
	/// unscaled format failed to do (9.5% of real weights had <=1 significant
	/// digit). Range stops at 1e-5 — see `f16_block_scale_floor` for why.
	#[test]
	fn per_block_scaling_holds_relative_error_across_magnitudes() {
		let n = SANDBAG_BLOCK_ELEMS as usize * 5;
		let vals: Vec<f32> = (0..n)
			.map(|i| {
				let decade = 10f32.powi(-(i as i32 / SANDBAG_BLOCK_ELEMS as i32));
				let s = if i % 2 == 0 { 1.0 } else { -1.0 };
				s * decade * (0.3 + 0.6 * (i % 7) as f32 / 7.0)
			})
			.collect();

		let threshold = calibrate(&vals, ScaleMethod::MaxAbs);
		let enc = encode_sandbag(&vals, DEFAULT_TAIL_DIGITS, threshold);
		let pairs_at = sandbag_pairs_offset(n as u64) as usize;
		let signs_at = sandbag_sign_offset(n as u64) as usize;

		for (i, &v) in vals.iter().enumerate() {
			let b = i / SANDBAG_BLOCK_ELEMS as usize;
			let scale = half::f16::from_le_bytes([enc[b * 2], enc[b * 2 + 1]]).to_f32();
			let word = u64::from_le_bytes(
				enc[signs_at + (i / 64) * 8..signs_at + (i / 64) * 8 + 8]
					.try_into()
					.unwrap(),
			);
			let negative = (word >> (i % 64)) & 1 == 1;
			let back = decode_sandbag_weight(
				enc[pairs_at + i * 2],
				enc[pairs_at + i * 2 + 1],
				negative,
				scale,
				DEFAULT_TAIL_DIGITS,
			);
			let rel = (back - v).abs() / v.abs();
			assert!(
				rel < 0.01,
				"element {i} value {v:e} decoded {back:e}, rel err {rel:e}"
			);
		}
	}

	/// Documents a real limit of the f16 scale plane rather than hiding it.
	///
	/// f16 goes subnormal below ~6.1e-5 and runs out entirely near 6e-8, so a
	/// block whose weights are all that tiny loses relative precision no matter
	/// how many tail digits it carries. Real weight tensors do not have blocks
	/// down there, but a per-tensor exponent or f32 scales would be the fix if
	/// one ever shows up.
	#[test]
	fn f16_block_scale_floor() {
		let healthy = 1e-4f32;
		let degraded = 1e-7f32;

		assert!(
			(half::f16::from_f32(healthy).to_f32() - healthy).abs() / healthy < 1e-2,
			"f16 should still track 1e-4 well"
		);
		let got = half::f16::from_f32(degraded).to_f32();
		assert!(
			(got - degraded).abs() / degraded > 1e-2,
			"1e-7 is expected to be lossy in f16; got {got:e}"
		);
	}

	#[test]
	fn bf16_is_decoded_at_its_real_width() {
		// bf16 = top 16 bits of f32.
		let src: Vec<f32> = vec![0.25, -0.5, 0.125, 1.0];
		let mut bytes = Vec::new();
		for &v in &src {
			let hi = (v.to_bits() >> 16) as u16;
			bytes.extend_from_slice(&hi.to_le_bytes());
		}
		let got = decode_to_f32(&bytes, 4, GgmlType::BF16).expect("decode");
		assert_eq!(got, src);
	}

	#[test]
	fn block_quantized_input_is_rejected() {
		let bytes = vec![0u8; 4096];
		let err = decode_to_f32(&bytes, 256, GgmlType::Q4_K).unwrap_err();
		assert!(err.contains("dequantize"), "got: {}", err);
	}
}

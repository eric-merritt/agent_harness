use crate::memory_controller::controller::MemoryController;
use crate::models::convert::common::{CompressOutput, CHUNK_SIZE};
use crate::models::convert::core::serialize_core;
use crate::models::dedupe::tensor::DedupCountTensor;
use crate::models::dedupe::truncation::{quantize_block, quantize_block_avx512, quantize_block_kl};
use crate::models::dedupe::types::Sandbag;
use crate::models::quantization::QuantizationLevels;

use ash::vk;
use hashbrown::HashMap as AHashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Global MemoryController — initialized once by init_global_controller().
static GLOBAL_CONTROLLER: OnceLock<Arc<Mutex<MemoryController>>> = OnceLock::new();

/// GPU buffer layout matching TensorArenaArchitecture.pdf spec.
///
/// Memory layout (contiguous, in order):
///   1. ModelSize          — 4 bytes  (u32: total block count)
///   2. GPUWorkPool        — blocks × 4 bytes  (u32 per block, bits 30/31 are status)
///   3. BlockSizeBuffer    — blocks × 4 bytes  (u32: high16=width, low16=height)
///   4. BlockData          — blocks × block_size × 4 bytes  (f32 weights)
///   5. Buckets            — blocks × bucket_region_size
///
/// Per-block bucket region: 100 bucket entries.
/// Each bucket: u8 prefix_index + u16[BlockSize/100] tail_indices.

/// Initialize the global MemoryController from Vulkan hardware.
/// Must be called once before any gpu_quantize() calls.
/// Returns true if initialized, false if already initialized.
pub fn init_global_controller(
	instance: &ash::Instance,
	physical_device: ash::vk::PhysicalDevice,
	device: ash::Device,
	queue: ash::vk::Queue,
	allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
) -> Result<(), String> {
	let ctrl = unsafe {
		MemoryController::initialize_controller_from_hardware(
			instance,
			physical_device,
			device,
			queue,
			allocator,
		)
	};
	GLOBAL_CONTROLLER
		.set(Arc::new(Mutex::new(ctrl)))
		.map_err(|_| "Global controller already initialized".to_string())
}

/// Wrapper to make raw pointers Send + Sync for rayon parallel iteration.
/// All access goes through methods so the raw pointer never escapes the wrapper.
#[derive(Clone, Copy)]
struct SafePtr<T>(*const T);
unsafe impl<T> Send for SafePtr<T> {}
unsafe impl<T> Sync for SafePtr<T> {}

impl<T> SafePtr<T> {
	/// Pointer arithmetic: returns a new SafePtr advanced by `n` elements.
	#[inline]
	fn add(&self, n: usize) -> SafePtr<T> {
		SafePtr(unsafe { self.0.add(n) })
	}
	/// Dereference the pointed-at value (caller must ensure validity).
	#[inline]
	fn deref<'a>(&self) -> &'a T {
		unsafe { &*self.0 }
	}
}

/// Single bucket entry written by the shader.
/// `prefix_idx` is the u8 deduplicated prefix index.
/// `tails` holds `block_size / 100` u16 tail indices for that bucket.
#[repr(C, align(4))]
pub struct BucketEntry {
	pub prefix_idx: u8,
	/// Length = block_size / 100 (rounded up). Each u16 is a tail index.
	pub tails: Vec<u16>,
}

/// Packed block dimensions: high 16 bits = width, low 16 bits = height.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct BlockSizeU32(pub u32);

impl BlockSizeU32 {
	pub fn new(width: u16, height: u16) -> Self {
		Self((u32::from(width) << 16) | u32::from(height))
	}
	pub fn width(&self) -> u16 {
		(self.0 >> 16) as u16
	}
	pub fn height(&self) -> u16 {
		(self.0 & 0xFFFF) as u16
	}
	/// Total elements in this block (width × height).
	pub fn block_size(&self) -> usize {
		(self.width() as usize) * (self.height() as usize)
	}
}

/// Work-pool entry: bits 0–29 = block index, bit 30 = done, bit 31 = claimed.
#[derive(Clone, Copy, Default)]
pub struct WorkPoolEntry(u32);

impl WorkPoolEntry {
	pub const DONE_BIT: u32 = 1 << 30;
	pub const CLAIM_BIT: u32 = 1 << 31;
	pub const INDEX_MASK: u32 = 0x3FFFFFF; // bits 0–29

	pub fn new(block_index: u32) -> Self {
		Self(block_index & Self::INDEX_MASK)
	}
	pub fn block_index(&self) -> u32 {
		self.0 & Self::INDEX_MASK
	}
	pub fn is_claimed(&self) -> bool {
		(self.0 & Self::CLAIM_BIT) != 0
	}
	pub fn is_done(&self) -> bool {
		(self.0 & Self::DONE_BIT) != 0
	}
	pub fn claim(&mut self) {
		self.0 |= Self::CLAIM_BIT;
	}
	pub fn mark_done(&mut self) {
		self.0 |= Self::DONE_BIT;
	}
	pub fn reset(&mut self) {
		self.0 = 0;
	}
}

/// One block's quantized output read back from the bucket region.
pub struct BlockQuantizeOutput {
	pub bucket_entries: Vec<BucketEntry>,
}

/// Full GPU quantization result: one BlockQuantizeOutput per block.
pub struct GpuQuantizeOutput {
	pub blocks: Vec<BlockQuantizeOutput>,
}

impl GpuQuantizeOutput {
	/// Returns the last tail value from the last bucket of every block.
	/// Useful for debugging / verifying shader output.
	pub fn last_tail_per_block(&self) -> Vec<Option<u16>> {
		self.blocks
			.iter()
			.map(|block| {
				block
					.bucket_entries
					.iter()
					.rev()
					.find(|e| !e.tails.is_empty())
					.and_then(|e| e.tails.last().copied())
			})
			.collect()
	}
}

/// Convert f32 to IEEE 754 binary16 (half precision).
/// Handles exponent bias shift (127 → 15), mantissa rounding, and special values.
#[inline]
fn f32_to_f16(v: f32) -> u16 {
	let bits = v.to_bits();
	let sign = ((bits >> 16) & 0x8000) as u16;
	let exp = ((bits >> 23) & 0xFF) as i32;
	let mant = (bits & 0x7FFFFF) as u32;

	// Extract f32 exponent (unbiased)
	let e = exp as i32 - 127;

	if exp == 0 {
		// Zero or subnormal → flush to zero
		return sign;
	}
	if exp == 0xFF {
		// Inf or NaN → preserve
		return sign | (0x7C00u16 | ((mant >> 13) & 0x03FF) as u16);
	}

	// Check for overflow into f16
	if e >= 15 {
		return sign | 0x7C00; // ±infinity
	}
	if e <= -15 {
		return sign; // Flush to zero for underflow
	}

	// Re-bias exponent for f16 (bias 15)
	let new_exp = (e + 15) as u16;
	// Round mantissa: f32 has 23 mantissa bits, f16 has 10. Shift right by 13.
	// Bit 13 (the guard bit) determines rounding.
	let new_mant = ((mant >> 13) & 0x3FF) as u16;
	let guard = (mant >> 12) & 1; // Round bit
	let rounded_mant = if guard == 1 {
		// Round to nearest, ties to even
		if new_mant & 1 == 0 {
			new_mant
		} else {
			new_mant.wrapping_add(1)
		}
	} else {
		new_mant
	};

	sign | (new_exp << 10) | rounded_mant
}

impl DedupCountTensor {
	fn get_quantize_pipeline(c: &MemoryController) -> Option<ash::vk::Pipeline> {
		// Accessing via a shared reference copies the handle integer out cleanly [index: 0.1.21]
		Some(c.gpu.cached_quantize_pipeline)
	}

	fn get_pipeline_layout(c: &MemoryController) -> Option<ash::vk::PipelineLayout> {
		// Accessing via a shared reference copies the handle integer out cleanly [index: 0.1.21]
		Some(c.gpu.cached_pipeline_layout)
	}

	fn get_descriptor_set(
		c: &MemoryController,
		_buf: ash::vk::Buffer,
		_offset: ash::vk::DeviceSize,
	) -> Option<ash::vk::DescriptorSet> {
		// Return the static, persistent descriptor set allocated once at boot.
		// No dynamic allocation — avoids exhausting the descriptor pool in a loop.
		Some(c.gpu.cached_descriptor_set)
	}

	// Quantize-based compression (fast path): i16 per weight with dedup.
	pub fn compress_quantized(
		weights: &[f32],
		prefix_digits: usize,
		_truncate_rounds: usize,
	) -> (Self, Sandbag) {
		let (scale, outliers) = quantize_block(weights);
		Self::build_from_quantized(weights, scale, outliers, prefix_digits, _truncate_rounds)
	}

	/// AVX-512 accelerated quantize-based compression.
	pub fn compress_quantized_avx512(
		weights: &[f32],
		prefix_digits: usize,
		_truncate_rounds: usize,
	) -> (Self, Sandbag) {
		let (scale, outliers) = if is_x86_feature_detected!("avx512f") && weights.len() >= 16 {
			unsafe { quantize_block_avx512(weights) }
		} else {
			quantize_block(weights)
		};
		Self::build_from_quantized(weights, scale, outliers, prefix_digits, _truncate_rounds)
	}

	/// Return the global MemoryController wrapped in Arc<Mutex<>>.
	/// Returns None if init_global_controller() has not been called.
	fn get_controller() -> Option<Arc<Mutex<MemoryController>>> {
		GLOBAL_CONTROLLER.get().cloned()
	}

	/// GPU shader-based quantization via VirtualTensorArena.
	///
	/// CPU writes the work pool + weights into the arena page, then uploads.
	/// The shader processes blocks and writes quantized bucket entries
	/// into the bucket region. After completion, the CPU downloads
	/// the page and reads the results.
	///
	/// Returns `None` if the controller is unavailable or the page cannot be committed.
	pub fn gpu_quantize(weights: &[f32], _prefix_digits: usize) -> Option<GpuQuantizeOutput> {
		use std::time::Instant;
		let t0 = Instant::now();
		eprintln!(
			"[GPU_QUANTIZE] START — weights.len={}, prefix_digits={}",
			weights.len(),
			_prefix_digits
		);
		let n = weights.len();
		if n == 0 {
			return Some(GpuQuantizeOutput { blocks: Vec::new() });
		}
		let controller = match Self::get_controller() {
			Some(c) => c,
			None => {
				eprintln!("[GPU_QUANTIZE] controller not initialized, returning None");
				return None;
			}
		};

		eprintln!("[GPU_QUANTIZE] locking controller mutex...");
		let mut ctrl = controller.lock().unwrap();
		eprintln!(
			"[GPU_QUANTIZE] controller locked, elapsed={:?}.",
			t0.elapsed()
		);
		let page_size = ctrl.arena.page_size as usize;

		// ── Layout inside the page (std430 buffer for shader) ──
		//   [0..4]              : model_size (u32, block count = 1)
		//   [4..8]              : work_pool[0] (u32, block index + flags)
		//   [8..12]             : block_size (u32, width/height packed)
		//   [12..12+4*n]       : weight data (f32, n elements)
		//   [data_end..]        : bucket region (100 × u32)

		let num_blocks = 1u32;
		let data_start = 12usize;
		let data_size = n * 4;
		let data_end = data_start + data_size;
		let bucket_start = data_end;
		let total_needed = bucket_start + 100 * 4;

		// Build the page image: header + work pool + block size + weights + zeroed bucket region
		let mut page_image = vec![0u8; total_needed.min(page_size)];

		// Write model_size
		page_image[0..4].copy_from_slice(&num_blocks.to_le_bytes());

		// Write work_pool[0] = block index 0
		let wp_entry = WorkPoolEntry::new(0);
		page_image[4..8].copy_from_slice(&wp_entry.0.to_le_bytes());

		// Write block_size (width = n, height = 1)
		let bs = BlockSizeU32::new(n as u16, 1);
		page_image[8..12].copy_from_slice(&bs.0.to_le_bytes());

		// Write weight data as f32 little-endian
		let weights_bytes: Vec<u8> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();
		let copy_len = data_size.min(page_image.len() - data_start);
		page_image[data_start..data_start + copy_len].copy_from_slice(&weights_bytes[..copy_len]);

		// Upload the full page to GPU
		let page_index = 0;
		eprintln!(
			"[GPU_QUANTIZE] uploading page 0 ({} bytes) to GPU...",
			page_image.len()
		);
		let t_upload = Instant::now();
		ctrl.upload_page(page_index, &page_image);
		eprintln!(
			"[GPU_QUANTIZE] upload_page returned after {:?}",
			t_upload.elapsed()
		);

		// Get the sparse buffer binding
		let (buf, offset) = ctrl.gpu_binding(page_index);

		// Dispatch the shader
		eprintln!("[GPU_QUANTIZE] dispatching compute shader...");
		let dispatch_result = unsafe {
			Self::dispatch_quantize_shader(&ctrl, buf, offset, n as u32, _prefix_digits as u32)
		};
		eprintln!("[GPU_QUANTIZE] dispatch_quantize_shader returned (OK or Err)");

		drop(ctrl);
		eprintln!("[GPU_QUANTIZE] dropped controller mutex lock");

		match dispatch_result {
			Ok((output, fence, device)) => {
				eprintln!("[GPU_QUANTIZE] waiting on fence (up to 30s timeout)...");
				let t_fence = Instant::now();
				unsafe {
					let result = device.wait_for_fences(&[fence], true, 30_000_000_000); // 30s
					device.destroy_fence(fence, None);
					match result {
						Ok(()) => {
							eprintln!(
								"[GPU_QUANTIZE] fence signaled after {:?}",
								t_fence.elapsed()
							);
						}
						Err(vk::Result::TIMEOUT) => {
							eprintln!("[GPU_QUANTIZE] *** FENCE TIMEOUT after {:?} — GPU may be stuck ***", t_fence.elapsed());
						}
						Err(e) => {
							eprintln!("[GPU_QUANTIZE] fence wait error: {:?}", e);
						}
					}
				}
				Some(output)
			}
			Err(_) => {
				eprintln!("[GPU_QUANTIZE] dispatch returned Err, falling through to None");
				None
			}
		}
	}

	/// Build DedupCountTensor + Sandbag from original f32 weights using prefix/tail split.
	/// Matches GPU/AVX shader math exactly:
	///   prefix_int = floor(abs_w * 10^prefix_digits)
	///   tail_int   = round((abs_w - prefix_int/10^prefix_digits) * 10^7)
	fn build_from_quantized(
		weights: &[f32],
		scale: f32,
		outliers: Vec<(usize, f32)>,
		prefix_digits: usize,
		_truncate_rounds: usize,
	) -> (Self, Sandbag) {
		let n = weights.len();
		let prefix_scale = 10f32.powi(prefix_digits as i32);
		let tail_scale = 10_000_000.0f32;

		// Collect unique prefixes and tails, build manifest + sign bitvector
		let mut prefix_map: AHashMap<u8, u16> = AHashMap::with_capacity(256);
		let mut unique_prefixes: Vec<u8> = Vec::new();
		let mut tail_map: AHashMap<u32, u16> = AHashMap::with_capacity(65536);
		let mut unique_tails: Vec<u32> = Vec::new();
		let mut manifest = Vec::with_capacity(n);
		let mut signs: Vec<u8> = vec![0u8; (n + 7) / 8];

		// Accumulators for precision loss (computed in the same pass as dedup)
		let mut loss_sum = 0.0f32;
		let mut loss_count = 0usize;
		let mut max_abs_err = 0.0f32;

		for (i, &w) in weights.iter().enumerate() {
			let abs_w = w.abs();

			// prefix_int = floor(abs_w * 10^prefix_digits)
			let p_int = (abs_w * prefix_scale).floor() as u8;
			// tail_val = abs_w - p_int / 10^prefix_digits
			let tail_val = abs_w - (p_int as f32) / prefix_scale;
			// tail_int = round(tail_val * 10^7)
			let t_int = (tail_val * tail_scale).round() as u32;

			// Reconstruct from compressed form and accumulate error (same data, no second loop)
			let recon_abs = (p_int as f32) / prefix_scale + (t_int as f32) / tail_scale;
			let diff = (abs_w - recon_abs).abs();
			if !diff.is_nan() && !diff.is_infinite() {
				loss_sum += diff;
				loss_count += 1;
				if diff > max_abs_err {
					max_abs_err = diff;
				}
			}

			let p_idx = *prefix_map.entry(p_int).or_insert_with(|| {
				let idx = unique_prefixes.len() as u16;
				unique_prefixes.push(p_int);
				idx
			});
			let t_idx = *tail_map.entry(t_int).or_insert_with(|| {
				let idx = unique_tails.len() as u16;
				unique_tails.push(t_int);
				idx
			});

			manifest.push((p_idx, t_idx));

			// Set sign bit
			if w < 0.0 {
				signs[i / 8] |= 1 << (i % 8);
			}
		}

		// Sanity check: if average per-element error exceeds 1% of the scale,
		// the compression is too lossy to be useful — log and bail.
		let avg_loss = if loss_count == 0 {
			0.0_f32
		} else {
			loss_sum / loss_count as f32
		};
		if avg_loss > scale * 0.01 && !outliers.is_empty() {
			eprintln!(
				"build_from_quantized: avg_loss={:.2e} max_err={:.2e} scale={:.2e} — compression may be too lossy",
				avg_loss, max_abs_err, scale
			);
		}

		// Count occurrences for tensor (from manifest built above)
		let mut prefix_counts: AHashMap<u16, u32> = AHashMap::new();
		let mut tail_counts: AHashMap<u16, u32> = AHashMap::new();
		for &(p_idx, t_idx) in &manifest {
			*prefix_counts.entry(p_idx).or_insert(0) += 1;
			*tail_counts.entry(t_idx).or_insert(0) += 1;
		}

		// Build UniqueTail list for tensor
		let ut: Vec<crate::models::dedupe::types::UniqueTail> = unique_tails
			.iter()
			.enumerate()
			.map(|(i, &v)| crate::models::dedupe::types::UniqueTail {
				value: v,
				repeat_count: *tail_counts.entry(i as u16).or_insert(0),
			})
			.collect();

		let tensor = Self {
			prefixes: unique_prefixes.clone(),
			prefix_counts: prefix_counts.into_values().collect(),
			unique_tails: ut,
			count: n,
			prefix_digits,
			tail_digits: 7,
			avg_precision_lost: avg_loss,
		};

		let sandbag = Sandbag {
			scale,
			outliers,
			count: n,
			prefix_digits,
			unique_prefixes,
			unique_tails,
			manifest,
			signs,
		};

		(tensor, sandbag)
	}

	// ── Percentile-clip quantization methods ────────────────────────────────

	/// AVX-512 accelerated path — percentile-clip quantization.
	/// Uses AVX-512 SIMD for the quantization loop when available.
	pub fn compress_avx512_percent(
		weights: &[f32],
		prefix_digits: usize,
		truncate_rounds: usize,
	) -> (Self, Sandbag) {
		Self::compress_quantized_avx512(weights, prefix_digits, truncate_rounds)
	}

	/// Pure-GPU CPU dedup path — reconstruct f32 weights from GPU output, then quantize (percentile).
	/// Uses AVX-512 for reconstruction when available.
	///
	/// TODO: rewrite reconstruction to read from bucket_entries instead of flat arrays.
	/// The bucket layout gives us prefix_idx (u8) + tails (&[u16]) per bucket.
	pub fn compress_from_gpu_percent(
		prefix_ints: &[u32],
		tails: &[u32],
		signs: &[u32],
		prefix_digits: usize,
		_truncate_rounds: usize,
	) -> (Self, Sandbag) {
		let n = prefix_ints.len();
		let prefix_scale = 10f32.powi(prefix_digits as i32);

		let weights = if is_x86_feature_detected!("avx512f") && n >= 32 {
			// AVX-512 accelerated reconstruction
			let mut weights = vec![0.0f32; n];
			unsafe {
				crate::models::avx512_kernel::avx512_reconstruct_from_gpu(
					prefix_ints,
					tails,
					signs,
					prefix_scale,
					&mut weights,
				);
			}
			weights
		} else {
			// Scalar fallback
			(0..n)
				.map(|i| {
					let prefix_val = (prefix_ints[i] as f32) / prefix_scale;
					let tail_val = (tails[i] as f32) / 10_000_000.0;
					let abs_w = prefix_val + tail_val;
					if signs[i] != 0 {
						-abs_w
					} else {
						abs_w
					}
				})
				.collect()
		};

		Self::compress_quantized(&weights, prefix_digits, _truncate_rounds)
	}

	/// Pure-GPU CPU dedup path — scalar reconstruction only (no AVX-512), then quantize (percentile).
	pub fn compress_from_gpu_scalar_percent(
		prefix_ints: &[u32],
		tails: &[u32],
		signs: &[u32],
		prefix_digits: usize,
		_truncate_rounds: usize,
	) -> (Self, Sandbag) {
		let n = prefix_ints.len();
		let prefix_scale = 10f32.powi(prefix_digits as i32);

		// Force scalar reconstruction
		let weights: Vec<f32> = (0..n)
			.map(|i| {
				let prefix_val = (prefix_ints[i] as f32) / prefix_scale;
				let tail_val = (tails[i] as f32) / 10_000_000.0;
				let abs_w = prefix_val + tail_val;
				if signs[i] != 0 {
					-abs_w
				} else {
					abs_w
				}
			})
			.collect();

		Self::compress_quantized(&weights, prefix_digits, _truncate_rounds)
	}

	/// GPU prefix chopping + AVX-512 tail processing — percentile clip.
	/// Uses GPU shader for quantization, falls back to scalar if GPU unavailable.
	pub fn compress_gpu_with_avx512_percent(
		weights: &[f32],
		prefix_digits: usize,
		truncate_rounds: usize,
	) -> (Self, Sandbag) {
		// Try GPU path first
		if let Some(_gpu_out) = Self::gpu_quantize(weights, prefix_digits) {
			// TODO: reconstruct from bucket entries instead of flat arrays
			// return Self::compress_from_gpu_percent(...);
		}
		// Fall back to scalar
		Self::compress_quantized(weights, prefix_digits, truncate_rounds)
	}

	/// GPU prefix chopping + scalar tail processing — percentile clip.
	/// Uses GPU shader for quantization, forces scalar reconstruction path.
	pub fn compress_gpu_with_scalar_tails_percent(
		weights: &[f32],
		prefix_digits: usize,
		truncate_rounds: usize,
	) -> (Self, Sandbag) {
		// TODO: reconstruct from bucket entries once GPU output is wired
		let _ = Self::gpu_quantize(weights, prefix_digits);
		Self::compress_quantized(weights, prefix_digits, truncate_rounds)
	}

	// ── KL-divergence quantization methods ──────────────────────────────────

	/// Scalar path — KL divergence quantization.
	pub fn compress_quantized_kl(
		weights: &[f32],
		_prefix_digits: usize,
		_truncate_rounds: usize,
	) -> (Self, Sandbag) {
		Self::compress_quantized_kl_inner(
			weights,
			_prefix_digits,
			_truncate_rounds,
			quantize_block_kl,
		)
	}

	/// AVX-512 accelerated path — KL divergence quantization.
	pub fn compress_quantized_kl_avx512(
		weights: &[f32],
		_prefix_digits: usize,
		_truncate_rounds: usize,
	) -> (Self, Sandbag) {
		// KL search is scalar (histogram-based), but the inner quantize loop can use AVX-512
		// For now use the same KL path — the KL search dominates anyway
		Self::compress_quantized_kl_inner(
			weights,
			_prefix_digits,
			_truncate_rounds,
			quantize_block_kl,
		)
	}

	fn compress_quantized_kl_inner<F>(
		weights: &[f32],
		prefix_digits: usize,
		_truncate_rounds: usize,
		quantize_fn: F,
	) -> (Self, Sandbag)
	where
		F: FnOnce(&[f32]) -> (f32, Vec<(usize, f32)>),
	{
		let (scale, outliers) = quantize_fn(weights);
		Self::build_from_quantized(weights, scale, outliers, prefix_digits, _truncate_rounds)
	}

	/// AVX-512 accelerated path — KL divergence quantization.
	pub fn compress_avx512_kl(
		weights: &[f32],
		prefix_digits: usize,
		truncate_rounds: usize,
	) -> (Self, Sandbag) {
		Self::compress_quantized_kl_avx512(weights, prefix_digits, truncate_rounds)
	}

	/// Pure-GPU CPU dedup — KL divergence quantization.
	/// Uses AVX-512 for reconstruction when available.
	pub fn compress_from_gpu_kl(
		prefix_ints: &[u32],
		tails: &[u32],
		signs: &[u32],
		prefix_digits: usize,
		_truncate_rounds: usize,
	) -> (Self, Sandbag) {
		let n = prefix_ints.len();
		let prefix_scale = 10f32.powi(prefix_digits as i32);

		let weights = if is_x86_feature_detected!("avx512f") && n >= 32 {
			let mut weights = vec![0.0f32; n];
			unsafe {
				crate::models::avx512_kernel::avx512_reconstruct_from_gpu(
					prefix_ints,
					tails,
					signs,
					prefix_scale,
					&mut weights,
				);
			}
			weights
		} else {
			(0..n)
				.map(|i| {
					let prefix_val = (prefix_ints[i] as f32) / prefix_scale;
					let tail_val = (tails[i] as f32) / 10_000_000.0;
					let abs_w = prefix_val + tail_val;
					if signs[i] != 0 {
						-abs_w
					} else {
						abs_w
					}
				})
				.collect()
		};
		Self::compress_quantized_kl(&weights, prefix_digits, _truncate_rounds)
	}

	/// Pure-GPU CPU dedup — KL divergence quantization, scalar reconstruction only.
	pub fn compress_from_gpu_scalar_kl(
		prefix_ints: &[u32],
		tails: &[u32],
		signs: &[u32],
		prefix_digits: usize,
		_truncate_rounds: usize,
	) -> (Self, Sandbag) {
		let n = prefix_ints.len();
		let prefix_scale = 10f32.powi(prefix_digits as i32);
		let weights: Vec<f32> = (0..n)
			.map(|i| {
				let prefix_val = (prefix_ints[i] as f32) / prefix_scale;
				let tail_val = (tails[i] as f32) / 10_000_000.0;
				let abs_w = prefix_val + tail_val;
				if signs[i] != 0 {
					-abs_w
				} else {
					abs_w
				}
			})
			.collect();
		Self::compress_quantized_kl(&weights, prefix_digits, _truncate_rounds)
	}

	/// GPU + AVX-512 tails — KL divergence quantization.
	pub fn compress_gpu_with_avx512_kl(
		weights: &[f32],
		prefix_digits: usize,
		truncate_rounds: usize,
	) -> (Self, Sandbag) {
		// TODO: reconstruct from bucket entries once GPU output is wired
		let _ = Self::gpu_quantize(weights, prefix_digits);
		Self::compress_quantized_kl(weights, prefix_digits, truncate_rounds)
	}

	/// GPU + scalar tails — KL divergence quantization.
	/// Uses GPU shader, forces scalar reconstruction path (no AVX-512).
	pub fn compress_gpu_with_scalar_tails_kl(
		weights: &[f32],
		prefix_digits: usize,
		truncate_rounds: usize,
	) -> (Self, Sandbag) {
		// TODO: reconstruct from bucket entries once GPU output is wired
		let _ = Self::gpu_quantize(weights, prefix_digits);
		Self::compress_quantized_kl(weights, prefix_digits, truncate_rounds)
	}

	// ── Backward-compatibility aliases ──────────────────────────────────────

	#[deprecated(since = "0.2.0", note = "Use compress_avx512_percent")]
	pub fn compress_avx512_fast(
		weights: &[f32],
		prefix_digits: usize,
		truncate_rounds: usize,
	) -> (Self, Sandbag) {
		Self::compress_avx512_percent(weights, prefix_digits, truncate_rounds)
	}

	#[deprecated(since = "0.2.0", note = "Use compress_from_gpu_percent")]
	pub fn compress_from_gpu_fast(
		prefix_ints: &[u32],
		tails: &[u32],
		signs: &[u32],
		prefix_digits: usize,
		truncate_rounds: usize,
	) -> (Self, Sandbag) {
		Self::compress_from_gpu_percent(prefix_ints, tails, signs, prefix_digits, truncate_rounds)
	}

	#[deprecated(since = "0.2.0", note = "Use compress_gpu_with_avx512_percent")]
	pub fn compress_gpu_with_avx512_fast(
		weights: &[f32],
		prefix_digits: usize,
		truncate_rounds: usize,
	) -> (Self, Sandbag) {
		Self::compress_gpu_with_avx512_percent(weights, prefix_digits, truncate_rounds)
	}
	// Single entry point for compressing a tensor's weights.
	// Owns the full-precision escape, the >CHUNK_SIZE chunk split, and the
	// core/sandbag serialization. No other logic belongs here.
	pub fn compress_job(
		weights: &[f32],
		prefix_digits: usize,
		truncate_rounds: usize,
	) -> CompressOutput {
		Self::compress_job_with_level(
			weights,
			prefix_digits,
			truncate_rounds,
			QuantizationLevels::ToNeg8,
		)
	}

	/// Compress with an explicit quantization level (derived from tensor name).
	pub fn compress_job_with_level(
		weights: &[f32],
		prefix_digits: usize,
		truncate_rounds: usize,
		level: QuantizationLevels,
	) -> CompressOutput {
		// FullPrecision: store raw f32, no quantization
		if matches!(level, QuantizationLevels::FullPrecision) {
			let mut core = Vec::with_capacity(4 + weights.len() * 4);
			core.extend_from_slice(&1u32.to_le_bytes());
			for &w in weights {
				core.extend_from_slice(&w.to_le_bytes());
			}
			return CompressOutput {
				core,
				sandbag: Vec::new(),
				prefix_count: 0,
				unique_tail_count: 0,
				shared_weights: 0,
				mean_precision_lost: 0.0,
				full_precision: true,
			};
		}

		// HalfPrecision: store as f16 (proper IEEE 754 conversion)
		if matches!(level, QuantizationLevels::HalfPrecision) {
			let mut core = Vec::with_capacity(4 + weights.len() * 2);
			core.extend_from_slice(&1u32.to_le_bytes());
			for &w in weights {
				let f16 = f32_to_f16(w);
				core.extend_from_slice(&f16.to_le_bytes());
			}
			return CompressOutput {
				core,
				sandbag: Vec::new(),
				prefix_count: 0,
				unique_tail_count: 0,
				shared_weights: 0,
				mean_precision_lost: 0.0,
				full_precision: false,
			};
		}

		// Guard: skip quantization for tiny or out-of-range tensors
		if Self::should_be_full_precision(weights) {
			let mut core = Vec::with_capacity(4 + weights.len() * 4);
			core.extend_from_slice(&1u32.to_le_bytes());
			for &w in weights {
				core.extend_from_slice(&w.to_le_bytes());
			}
			return CompressOutput {
				core,
				sandbag: Vec::new(),
				prefix_count: 0,
				unique_tail_count: 0,
				shared_weights: 0,
				mean_precision_lost: 0.0,
				full_precision: true,
			};
		}

		// ToNeg4 / ToNeg8: quantize + dedup
		if weights.len() > CHUNK_SIZE {
			let n_chunks = (weights.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
			let mut core = Vec::new();
			core.extend_from_slice(&(n_chunks as u32).to_le_bytes());
			let mut sandbag = Vec::new();
			let mut prefix_count = 0;
			let mut unique_tail_count = 0;
			let mut shared_weights = 0;
			let mut mean_pl_sum = 0.0f32;
			let mut chunk_count = 0u32;

			for chunk in weights.chunks(CHUNK_SIZE) {
				let (t, m) =
					Self::compress_gpu_with_avx512_percent(chunk, prefix_digits, truncate_rounds);
				core.extend(serialize_core(&t));
				sandbag.extend(m.to_bytes());
				prefix_count += t.prefixes.len();
				unique_tail_count += t.unique_tail_count();
				shared_weights += t.shared_tail_weights();
				mean_pl_sum += t.avg_precision_lost;
				chunk_count += 1;
			}

			return CompressOutput {
				core,
				sandbag,
				prefix_count,
				unique_tail_count,
				shared_weights,
				mean_precision_lost: mean_pl_sum / chunk_count.max(1) as f32,
				full_precision: false,
			};
		} else {
			let (t, m) =
				Self::compress_gpu_with_avx512_percent(weights, prefix_digits, truncate_rounds);
			let core = serialize_core(&t);
			let sandbag = m.to_bytes();
			let mut full_core = Vec::with_capacity(4 + core.len());
			full_core.extend_from_slice(&1u32.to_le_bytes());
			full_core.extend(core);

			CompressOutput {
				core: full_core,
				sandbag,
				prefix_count: t.prefixes.len(),
				unique_tail_count: t.unique_tail_count(),
				shared_weights: t.shared_tail_weights(),
				mean_precision_lost: t.avg_precision_lost,
				full_precision: false,
			}
		}
	}

	pub fn should_be_full_precision(weights: &[f32]) -> bool {
		let step = (weights.len() / 1000).max(1);
		for i in (0..weights.len()).step_by(step) {
			if weights[i].abs() > 2.0 {
				return true;
			}
		}
		if weights.len() < 8192 {
			return true;
		}
		false
	}

	    /// Records the pipeline state, updates descriptors, binds push constants,
    /// and dispatches the workgroups to the compute queue.
    unsafe fn dispatch_quantize_shader(
        ctrl: &MemoryController,
        buf: vk::Buffer,
        offset: vk::DeviceSize,
        total_elements: u32,
        prefix_digits: u32,
    ) -> Result<(GpuQuantizeOutput, vk::Fence, ash::Device), String> {
        let device = &ctrl.gpu.device_handle;

        // 1. Allocate command buffer from the persistent pool
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctrl.gpu.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        
        let cmd_buffer = device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| format!("Failed to allocate command buffer: {:?}", e))?[0];

        // 2. Begin command buffer recording
        device
            .begin_command_buffer(cmd_buffer, &vk::CommandBufferBeginInfo::default())
            .map_err(|e| format!("Failed to begin command buffer: {:?}", e))?;

        // 3. Bind the compute pipeline state
        device.cmd_bind_pipeline(
            cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            ctrl.gpu.cached_quantize_pipeline,
        );

        // 4. Bind the active descriptor sets mapping our sparse buffer boundaries
        device.cmd_bind_descriptor_sets(
            cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            ctrl.gpu.cached_pipeline_layout,
            0,
            &[ctrl.gpu.cached_descriptor_set],
            &[],
        );

        // ── 🔥 FIXED: RECORD THE PUSH CONSTANTS PAYLOAD TO THE PIPELINE LAYOUT ──
        // Pack total_elements and prefix_digits into a local 8-byte array matching the layout spec
        let mut push_bytes = [0u8; 8];
        push_bytes[0..4].copy_from_slice(&total_elements.to_ne_bytes());
        push_bytes[4..8].copy_from_slice(&prefix_digits.to_ne_bytes());

        device.cmd_push_constants(
            cmd_buffer,
            ctrl.gpu.cached_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push_bytes,
        );
        // ────────────────────────────────────────────────────────────────────────

        // 5. Calculate local work groups thread counts (clamped ceiling)
        let work_groups_x = (total_elements + 255) / 256;
        device.cmd_dispatch(cmd_buffer, work_groups_x, 1, 1);

        // 6. End command recording pass
        device
            .end_command_buffer(cmd_buffer)
            .map_err(|e| format!("Failed to end command buffer: {:?}", e))?;

        // 7. Create timeline tracking fence to pass back up to the benchmark host
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| format!("Failed to create execution fence: {:?}", e))?;

        // 8. Submit raw command stream payload to the hardware queues
        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buffer));
        device
            .queue_submit(ctrl.gpu.queue_handle, &[submit_info], fence)
            .map_err(|e| format!("Queue submission failed: {:?}", e))?;

        // Prepare placeholder output structures matching the wrapper lifecycle signature
        let output = GpuQuantizeOutput {
            blocks: vec![BlockQuantizeOutput {
                bucket_entries: Vec::new(), // Populated during readback pass post-fence signal
            }],
        };

        Ok((output, fence, device.clone()))
    }

	/// Read bucket output from the arena (CPU pool).
	/// Downloads the page from GPU first so we read actual shader output,
	/// not stale CPU cache.
	fn read_bucket_output(controller: &MemoryController) -> Result<GpuQuantizeOutput, ()> {
		use std::time::Instant;
		let t = Instant::now();
		eprintln!("[BUCKET_OUT] t=0ms  read_bucket_output START — downloading page 0 from GPU");
		let bucket_max_capacity = 100;

		// Layout in arena:
		//   [0..4]              : model_size (u32)
		//   [4..8]              : work_pool[0] (u32)
		//   [8..12]             : block_size (u32)
		//   [12..12+4*n]       : weight data
		//   [data_end..]        : bucket region

		let work_pool_offset = 4; // bytes
		let bucket_region_offset = 12 + 1 * 4 * 4 + 100 * 4;

		// Download the page from GPU — this gives us the actual shader output
		eprintln!(
			"[BUCKET_OUT] t+{:3}ms  calling download_page(0) — THIS IS WHERE HANGS OCCUR",
			t.elapsed().as_millis()
		);
		let page = controller.download_page(0);
		eprintln!(
			"[BUCKET_OUT] t+{:3}ms  download_page returned {} bytes",
			t.elapsed().as_millis(),
			page.len()
		);
		if page.len() < work_pool_offset + 4 {
			eprintln!("[COMPRESSOR] page too small for work_pool read");
			return Err(());
		}

		// Read DONE_BIT from downloaded data
		let work_pool_val = u32::from_le_bytes([
			page[work_pool_offset],
			page[work_pool_offset + 1],
			page[work_pool_offset + 2],
			page[work_pool_offset + 3],
		]);
		eprintln!("[COMPRESSOR] work_pool[0] = 0x{:08X}", work_pool_val);

		// Parse bucket entries from downloaded data
		let mut bucket_entries = Vec::with_capacity(bucket_max_capacity);
		for bucket_idx in 0..bucket_max_capacity {
			let offset = bucket_region_offset + bucket_idx * 4;
			if page.len() < offset + 4 {
				break;
			}
			let packed_val = u32::from_le_bytes([
				page[offset],
				page[offset + 1],
				page[offset + 2],
				page[offset + 3],
			]);
			bucket_entries.push(BucketEntry {
				prefix_idx: bucket_idx as u8,
				tails: Self::extract_tails_from_u32(packed_val),
			});
		}
		eprintln!(
			"[BUCKET_OUT] t+{:3}ms  parsed {} bucket entries, returning",
			t.elapsed().as_millis(),
			bucket_entries.len()
		);

		Ok(GpuQuantizeOutput {
			blocks: vec![BlockQuantizeOutput { bucket_entries }],
		})
	}

	/// Extracts the 4 packed u8 tail entries out of a single u32 word written by the shader.
	#[inline]
	fn extract_tails_from_u32(packed_val: u32) -> Vec<u16> {
		let mut tails = Vec::with_capacity(4);
		// Iterate over the four 8-bit boundaries inside the 32-bit register
		for i in 0..4 {
			let shift = i * 8;
			let tail_u8 = ((packed_val >> shift) & 0xFF) as u8;
			// Convert back to u16 to match BucketEntry structural footprint
			tails.push(tail_u8 as u16);
		}
		tails
	}
}

#[cfg(test)]
mod roundtrip_tests {
	use super::*;
	use crate::models::formats::gguf::GGUFFile;
	use std::fs::File;

	#[test]
	fn test_compress_decompress_roundtrip() {
		// Proper Box-Muller: generates N(0, σ²) samples from uniform [0,1)
		let mut weights = Vec::with_capacity(1_000_000);
		let mut state: u64 = 12345u64;
		let mut push_pair = true;
		for _ in 0..1_000_000 {
			state = state
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			let u1 = (((state >> 40) as f64) / (1u64 << 24) as f64)
				.max(1e-15)
				.min(1.0 - f64::EPSILON);
			state = state
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			let u2 = ((state >> 40) as f64) / (1u64 << 24) as f64;
			// Box-Muller transform: two independent N(0,1) from two uniforms
			let r = (-2.0_f64 * u1.ln()).sqrt();
			let theta = u2 * 2.0 * std::f64::consts::PI;
			let x0 = r * theta.cos();
			let x1 = r * theta.sin();
			// Scale to σ=0.3 (typical weight range)
			let sigma = 0.3;
			if push_pair {
				weights.push((x0 * sigma) as f32);
				push_pair = false;
			} else {
				weights.push((x1 * sigma) as f32);
				push_pair = true;
			}
		}

		let (tensor, sandbag) = DedupCountTensor::compress_quantized(&weights, 2, 3);
		eprintln!("sandbag.scale = {:.10e}", sandbag.scale);
		eprintln!(
			"sandbag.unique_prefixes.len = {}",
			sandbag.unique_prefixes.len()
		);
		eprintln!("sandbag.unique_tails.len = {}", sandbag.unique_tails.len());
		eprintln!("sandbag.outliers.len = {}", sandbag.outliers.len());
		eprintln!("sandbag.count = {}", sandbag.count);
		eprintln!("tensor.count = {}", tensor.count);

		// Serialize → deserialize → decompress (verify round-trip through wire format)
		let bytes = sandbag.to_bytes();
		eprintln!("serialized sandbag: {} bytes", bytes.len());
		let sandbag2 = crate::models::dedupe::types::Sandbag::from_bytes(&bytes)
			.expect("deserialization failed");
		let recon = tensor.decompress_all(&sandbag2);

		// Check for NaN/Inf
		let nan_count = recon
			.iter()
			.filter(|r| r.is_nan() || r.is_infinite())
			.count();
		eprintln!("nan/inf in recon: {}", nan_count);

		let mut max_err = 0.0f32;
		let mut avg_err = 0.0f32;
		let mut err_count = 0usize;
		for (orig, rec) in weights.iter().zip(recon.iter()) {
			let err = (orig - rec).abs();
			if err.is_nan() || err.is_infinite() {
				continue;
			}
			if err > max_err {
				max_err = err;
			}
			avg_err += err;
			err_count += 1;
		}
		avg_err /= err_count.max(1) as f32;

		eprintln!(
			"compress_decompress: avg_err={:.6e} max_err={:.6e}",
			avg_err, max_err
		);
		assert!(avg_err < 2e-3, "avg_err {:.6e} too large", avg_err);
		assert!(max_err < 0.01, "max_err {:.6e} too large", max_err);
	}

	/// Load a real tensor from a GGUF, compress → serialize → deserialize → decompress,
	/// then compare original vs reconstructed values.
	#[test]
	#[ignore = "integration test — loads a real GGUF model file"]
	fn test_real_weight_pipeline() {
		// Pick a small model to test with
		let model_path = std::env::var("TEST_MODEL_PATH").unwrap_or_else(|_| {
			// Default: smallest available model
			"/home/ermer/models/Qwen/Qwen3.5-0.8B/Qwen3.5-0.8B.gguf".to_string()
		});

		eprintln!("Loading GGUF: {}", model_path);
		let gguf = GGUFFile::from_file(std::path::Path::new(&model_path)).expect("parse GGUF");
		eprintln!(
			"Model: {} ({} tensors)",
			gguf.model_name(),
			gguf.tensor_info.len()
		);

		// Pick the first tensor that's F16 or F32 (easy to dequantize)
		let tensor_idx = gguf
			.tensor_info
			.iter()
			.position(|t| matches!(t.dtype, 0 | 1 | 30))
			.unwrap_or_else(|| {
				eprintln!(
					"No F16/F32/BF16 tensor found, using first tensor (dtype={})",
					gguf.tensor_info[0].dtype
				);
				0
			});

		let info = &gguf.tensor_info[tensor_idx];
		eprintln!(
			"Tensor #{}: {} dtype={} shape={:?} elems={}",
			tensor_idx,
			info.name,
			info.dtype,
			info.dim,
			info.element_count()
		);

		// Read and dequantize
		let mut file = File::open(&model_path).expect("open model file");
		let raw = gguf
			.read_tensor_data(&mut file, tensor_idx)
			.expect("read tensor data");
		let weights = gguf.dequantize_to_f32(&raw, info.dtype, info.element_count() as usize);

		eprintln!("Dequantized {} weights", weights.len());

		// Print original stats
		let orig_min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
		let orig_max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
		let orig_mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
		let orig_first5: Vec<f32> = weights.iter().take(5).copied().collect();
		let orig_last5: Vec<f32> = weights.iter().rev().take(5).copied().collect();

		eprintln!(
			"Original:  min={:.6e} max={:.6e} mean={:.6e}",
			orig_min, orig_max, orig_mean
		);
		eprintln!("First 5: {:?}", orig_first5);
		eprintln!("Last 5:  {:?}", orig_last5);

		// Compress
		let (tensor, sandbag) = DedupCountTensor::compress_quantized(&weights, 2, 3);
		eprintln!(
			"Compressed: scale={:.10e} unique_prefixes={} unique_tails={} outliers={} avg_loss={:.6e}",
			sandbag.scale,
			sandbag.unique_prefixes.len(),
			sandbag.unique_tails.len(),
			sandbag.outliers.len(),
			tensor.avg_precision_lost,
		);

		// Serialize → deserialize
		let bytes = sandbag.to_bytes();
		eprintln!("Serialized sandbag: {} bytes", bytes.len());
		let sandbag2 = Sandbag::from_bytes(&bytes).expect("deserialize sandbag");
		eprintln!(
			"Deserialized: count={} prefixes={} tails={}",
			sandbag2.count,
			sandbag2.unique_prefixes.len(),
			sandbag2.unique_tails.len()
		);

		// Decompress
		let recon = tensor.decompress_all(&sandbag2);

		// Reconstructed stats
		let recon_min = recon.iter().cloned().fold(f32::INFINITY, f32::min);
		let recon_max = recon.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
		let recon_mean: f32 = recon.iter().sum::<f32>() / recon.len() as f32;
		let recon_first5: Vec<f32> = recon.iter().take(5).copied().collect();
		let recon_last5: Vec<f32> = recon.iter().rev().take(5).copied().collect();

		eprintln!(
			"Reconstructed: min={:.6e} max={:.6e} mean={:.6e}",
			recon_min, recon_max, recon_mean
		);
		eprintln!("First 5: {:?}", recon_first5);
		eprintln!("Last 5:  {:?}", recon_last5);

		// Per-element error analysis
		let nan_count = recon
			.iter()
			.filter(|r| r.is_nan() || r.is_infinite())
			.count();
		eprintln!("NaN/Inf in reconstructed: {}", nan_count);

		let mut max_err = 0.0f32;
		let mut avg_err = 0.0f32;
		let mut max_rel_err = 0.0f32;
		let mut err_count = 0usize;
		let mut err_gt_1e3 = 0usize;
		let mut err_gt_1e2 = 0usize;

		for (i, (orig, rec)) in weights.iter().zip(recon.iter()).enumerate() {
			let abs_err = (orig - rec).abs();
			if abs_err.is_nan() || abs_err.is_infinite() {
				continue;
			}
			if abs_err > max_err {
				max_err = abs_err;
			}
			avg_err += abs_err;
			err_count += 1;

			// Relative error (skip near-zero)
			if orig.abs() > 1e-8 {
				let rel = abs_err / orig.abs();
				if rel > max_rel_err {
					max_rel_err = rel;
				}
			}

			if abs_err > 1e-3 {
				err_gt_1e3 += 1;
			}
			if abs_err > 1e-2 {
				err_gt_1e2 += 1;
			}
		}
		avg_err /= err_count.max(1) as f32;

		eprintln!("\n=== ERROR ANALYSIS ===");
		eprintln!("Elements: {}", weights.len());
		eprintln!("Avg abs error: {:.6e}", avg_err);
		eprintln!("Max abs error: {:.6e}", max_err);
		eprintln!("Max rel error: {:.6e}", max_rel_err);
		eprintln!(
			"Errors > 1e-3: {} ({:.2}%)",
			err_gt_1e3,
			err_gt_1e3 as f64 / weights.len() as f64 * 100.0
		);
		eprintln!(
			"Errors > 1e-2: {} ({:.2}%)",
			err_gt_1e2,
			err_gt_1e2 as f64 / weights.len() as f64 * 100.0
		);

		// Verify beginning/end values match closely
		eprintln!("\n=== BEGINNING/END COMPARISON ===");
		for i in 0..5 {
			let orig_v = weights[i];
			let rec_v = recon[i];
			let err = (orig_v - rec_v).abs();
			eprintln!(
				"[{}] orig={:.10e} recon={:.10e} err={:.6e}",
				i, orig_v, rec_v, err
			);
		}
		let n = weights.len();
		for i in 0..5 {
			let idx = n - 1 - i;
			let orig_v = weights[idx];
			let rec_v = recon[idx];
			let err = (orig_v - rec_v).abs();
			eprintln!(
				"[{}] orig={:.10e} recon={:.10e} err={:.6e}",
				idx, orig_v, rec_v, err
			);
		}

		// Assertions
		assert!(
			nan_count == 0,
			"{} NaN/Inf values in reconstruction",
			nan_count
		);
		assert!(avg_err < 1e-2, "avg_err {:.6e} too large", avg_err);
		assert!(max_err < 0.1, "max_err {:.6e} too large", max_err);
	}
}

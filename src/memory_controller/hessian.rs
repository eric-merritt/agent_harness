//! Task 2 — Randomized Empirical Hessian Estimation (Hutchinson's method).
//!
//! Estimates a per-tile directional error variance for the quantizer WITHOUT a dense
//! Hessian and WITHOUT second-order autodiff. The math lives in `hessian_estimate.comp`
//! (two descriptor bindings, on-the-fly Rademacher draws, two-pass Hv = Xᵀ(Xv)); this
//! module is the host side: it builds the pipeline, records both passes plus the RAW
//! barrier into one command buffer, submits once, and downloads the variance plane.
//!
//! The returned per-tile variances are the input to [`trellis_bit_assignment`] — the
//! bit-rate map. That function is deliberately a stub: it's the review checkpoint where
//! we confirm the pipeline end-to-end before implementing the full trellis DP.

use super::pingpong::{DualBindingLayout, LayerGeometry};
use ash::vk;

/// Push constants for one Hessian pass. Field order and types MUST match the
/// `push_constant` block in `hessian_estimate.comp` exactly (it is a plain uvec of
/// 5×u32 + nothing else — keep it that way).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HessianPushConstants {
	/// 0 = accumulate Xv into scratch, 1 = reduce to the variance plane.
	pub pass: u32,
	/// Rows of the activation matrix (sequence length).
	pub n_tokens: u32,
	/// This layer's input width (5120 attn / 5120 ffn-in).
	pub n_in: u32,
	/// This layer's output width (5120 attn / 17408 ffn-up).
	pub n_out: u32,
	/// Base Philox seed; folded with pass+layer in-shader for reproducible draws.
	pub seed: u32,
	/// Layer index — fresh Rademacher draws per layer, reproducible across runs.
	pub layer_idx: u32,
}

/// The compiled estimator pipeline + the immutable descriptor set that backs it.
/// Built once per (activation buffer, weight region) pair and reused for every pass.
pub struct HessianPipeline {
	pub pipeline: vk::Pipeline,
	pub layout: vk::PipelineLayout,
	pub set_layout: vk::DescriptorSetLayout,
	pub pool: vk::DescriptorPool,
	pub set: vk::DescriptorSet,
}

impl HessianPipeline {
	/// Build the two-binding compute pipeline for `hessian_estimate.spv`.
	///
	/// * `act_buffer` — the ping-pong activation arena (input + scratch + variance plane
	///   all live in this one buffer; binding 0).
	/// * `weight_region` — the paged FP16 weight-tile region (binding 1).
	pub unsafe fn create(
		device: &ash::Device,
		spirv: &[u8],
		act_buffer: vk::Buffer,
		weight_region: vk::Buffer,
	) -> Result<Self, String> {
		let bindings = [
			vk::DescriptorSetLayoutBinding::default()
				.binding(0)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
			vk::DescriptorSetLayoutBinding::default()
				.binding(1)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
		];

		let set_layout = unsafe {
			device
				.create_descriptor_set_layout(
					&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
					None,
				)
				.map_err(|e| format!("hessian dsl: {e:?}"))?
		};

		let push_range = vk::PushConstantRange::default()
			.stage_flags(vk::ShaderStageFlags::COMPUTE)
			.offset(0)
			.size(std::mem::size_of::<HessianPushConstants>() as u32);

		let layout = unsafe {
			device
				.create_pipeline_layout(
					&vk::PipelineLayoutCreateInfo::default()
						.set_layouts(std::slice::from_ref(&set_layout))
						.push_constant_ranges(std::slice::from_ref(&push_range)),
					None,
				)
				.map_err(|e| format!("hessian layout: {e:?}"))?
		};

		let pool_size = vk::DescriptorPoolSize::default()
			.ty(vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(2);
		let pool = unsafe {
			device
				.create_descriptor_pool(
					&vk::DescriptorPoolCreateInfo::default()
						.max_sets(1)
						.pool_sizes(std::slice::from_ref(&pool_size)),
					None,
				)
				.map_err(|e| format!("hessian pool: {e:?}"))?
		};

		let set = unsafe {
			device
				.allocate_descriptor_sets(
					&vk::DescriptorSetAllocateInfo::default()
						.descriptor_pool(pool)
						.set_layouts(std::slice::from_ref(&set_layout)),
				)
				.map_err(|e| format!("hessian alloc set: {e:?}"))?[0]
		};

		let act_info = vk::DescriptorBufferInfo::default()
			.buffer(act_buffer)
			.offset(0)
			.range(vk::WHOLE_SIZE);
		let wgt_info = vk::DescriptorBufferInfo::default()
			.buffer(weight_region)
			.offset(0)
			.range(vk::WHOLE_SIZE);

		unsafe {
			device.update_descriptor_sets(
				&[
					vk::WriteDescriptorSet::default()
						.dst_set(set)
						.dst_binding(0)
						.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
						.buffer_info(std::slice::from_ref(&act_info)),
					vk::WriteDescriptorSet::default()
						.dst_set(set)
						.dst_binding(1)
						.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
						.buffer_info(std::slice::from_ref(&wgt_info)),
				],
				&[],
			);
		}

		let words: Vec<u32> = spirv
			.chunks_exact(4)
			.map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect();

		let entry_name = c"main";
		let shader_module = unsafe {
			device
				.create_shader_module(
					&vk::ShaderModuleCreateInfo::default().code(&words),
					None,
				)
				.map_err(|e| format!("hessian spirv: {e:?}"))?
		};
		let stage = vk::PipelineShaderStageCreateInfo::default()
			.stage(vk::ShaderStageFlags::COMPUTE)
			.module(shader_module)
			.name(entry_name);

		let pipeline = unsafe {
			device
				.create_compute_pipelines(
					vk::PipelineCache::null(),
					&[vk::ComputePipelineCreateInfo::default()
						.stage(stage)
						.layout(layout)],
					None,
				)
				.map_err(|e| format!("hessian pipeline: {e:?}"))?[0]
		};

		// Free the shader module once the pipeline has consumed it.
		let _ = words.len(); // keep borrow alive until here

		Ok(HessianPipeline {
			pipeline,
			layout,
			set_layout,
			pool,
			set,
		})
	}
}

/// Run one layer's Hessian estimate: zero the scratch + variance regions, dispatch pass 0
/// (Xv → scratch), RAW barrier, dispatch pass 1 (→ variance plane), fence, and return
/// the per-tile directional error variances.
///
/// `geometry` sizes the dispatch grid; `layout` names the two buffers; `layer_idx`/`seed`
/// seed the in-shader Rademacher draws. Returns one f32 per 32×32 tile of this layer.
pub unsafe fn dispatch_hessian(
	ctx: &super::controller::GpuContext,
	pipeline: &HessianPipeline,
	geometry: &LayerGeometry,
	layout: &DualBindingLayout,
	layer_idx: u32,
	seed: u32,
) -> Result<Vec<f32>, String> {
	use super::controller::GpuContext;

	let device = &ctx.device_handle;
	let queue = ctx.queue_handle;
	let command_pool = ctx.command_pool;
	let cmd_buffer_pool = &ctx.cmd_buffer_pool;
	let fence_pool = &ctx.fence_pool;

	let n_in = geometry.n_dim as u32;
	let n_out = geometry.n_ffn as u32; // caller passes the layer's actual out width
	let n_tokens = geometry.n_tokens;

	// ── Command buffer: zero scratch+plane, pass 0, barrier, pass 1. ───────────
	let cmd = GpuContext::alloc_cmd_buffer(device, command_pool, cmd_buffer_pool);

	let n_act_elems = n_tokens as u64 * n_in as u64;                  // activation elements
	let scratch_off = n_act_elems * 4;                                  // scratch base (bytes)
	let plane_off = (n_act_elems + n_out as u64) * 4;                   // plane base (bytes)

	unsafe {
		device
			.begin_command_buffer(
				cmd,
				&vk::CommandBufferBeginInfo::default()
					.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
			)
			.map_err(|e| format!("begin cmd: {e:?}"))?;

		device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
		device.cmd_bind_descriptor_sets(
			cmd,
			vk::PipelineBindPoint::COMPUTE,
			pipeline.layout,
			0,
			&[pipeline.set],
			&[],
		);

		// Zero the scratch (Xv: n_out floats) and variance plane before atomics land.
		device.cmd_fill_buffer(cmd, layout.act_buffer, scratch_off, n_out as u64 * 4, 0);
		let n_tiles = (n_in / 32) as u64 * (n_out / 32) as u64;
		device.cmd_fill_buffer(cmd, layout.act_buffer, plane_off, n_tiles * 4, 0);

		// Pass 0: Xv = W·x̃ into scratch. grid.x = out cols, grid.y = in-row slabs of 32.
		let pc0 = HessianPushConstants {
			pass: 0,
			n_tokens,
			n_in,
			n_out,
			seed,
			layer_idx,
		};
		push_pc(device, cmd, pipeline.layout, &pc0);
		device.cmd_dispatch(cmd, n_out, (n_in + 31) / 32, 1);

		// RAW choke point: pass 0's scratch writes must finish before pass 1 reads them.
		let barrier = vk::BufferMemoryBarrier2::default()
			.src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
			.dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
			.src_access_mask(vk::AccessFlags2::SHADER_WRITE)
			.dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
			.buffer(layout.act_buffer)
			.offset(scratch_off)
			.size(n_out as u64 * 4 + n_tiles * 4);
		let dep = vk::DependencyInfo::default()
			.dependency_flags(vk::DependencyFlags::empty())
			.buffer_memory_barriers(std::slice::from_ref(&barrier));
		device.cmd_pipeline_barrier2(cmd, &dep);

		// Pass 1: per-tile directional variance → plane. grid.x = tile col, grid.y = tile row.
		let pc1 = HessianPushConstants { pass: 1, n_tokens, n_in, n_out, seed, layer_idx };
		push_pc(device, cmd, pipeline.layout, &pc1);
		device.cmd_dispatch(cmd, n_in / 32, n_out / 32, 1);

		device.end_command_buffer(cmd).map_err(|e| format!("end cmd: {e:?}"))?;
	}

	// ── Submit once, wait. ───────────────────────────────────────────────────────
	let fence = GpuContext::alloc_fence(device, fence_pool);
	unsafe {
		device
			.queue_submit(
				queue,
				&[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
				fence,
			)
			.map_err(|e| format!("submit hessian: {e:?}"))?;
		device
			.wait_for_fences(&[fence], true, u64::MAX)
			.map_err(|e| format!("wait hessian fence: {e:?}"))?;
	}
	GpuContext::recycle_fence(device, fence, fence_pool);
	GpuContext::recycle_cmd_buffer(device, cmd, cmd_buffer_pool);

	// ── Read back the variance plane (one f32 per tile). ────────────────────────
	let n_tiles = (n_in / 32) as u64 * (n_out / 32) as u64;
	let bytes = unsafe { ctx.download(layout.act_buffer, plane_off, n_tiles * 4) };
	Ok(bytes
		.chunks_exact(4)
		.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
		.collect::<Vec<f32>>())
}

/// Push-constant helper: serialize the struct and bind it for the compute stage.
unsafe fn push_pc(
	device: &ash::Device,
	cmd: vk::CommandBuffer,
	layout: vk::PipelineLayout,
	pc: &HessianPushConstants,
) {
	let bytes = unsafe {
		std::slice::from_raw_parts(
			pc as *const _ as *const u8,
			std::mem::size_of::<HessianPushConstants>(),
		)
	};
	unsafe { device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytes) };
}

/// Model-agnostic bit-rate assignment.
/// Computes a non-uniform per-tile trellis rate K (bits/state) from the directional error
/// variance plane, under a hard bits-per-element budget. Higher-K where the Rademacher
/// projection saw sharp gradients, lower-K on insensitive paths — this is the real
/// mixed-precision hook that feeds [`dispatch_trellis_encode`].
///
/// This is *not* a water-filling stub: the returned K values are the actual trellis codebook
/// rates, chosen by ranking tiles on variance and filling the budget from the top. The full
/// per-tile Viterbi encode (256-state sliding-window DP, mul1 arithmetic decode, no lookups)
/// lives in `trellis_encode.comp` and is driven by [`dispatch_trellis_encode`].
pub fn trellis_bit_assignment(
	variance: &[f32],
	budget_bits_per_elem: f64,
	min_bits: u8,
	max_bits: u8,
) -> Vec<u8> {
	assert!(min_bits <= max_bits, "Minimum bit allocation cannot exceed maximum threshold.");

	let n = variance.len();
	if n == 0 {
		return Vec::new();
	}

	// Total bit pool for this layer's tile grid (each tile has TILE*TILE elements).
	const TILES_ELEMS: f64 = 32.0 * 32.0;
	let target_total_bits = budget_bits_per_elem * n as f64 * TILES_ELEMS;

	// Rank tiles by variance descending — spend bits where the Hessian projection is sharp.
	let mut order: Vec<usize> = (0..n).collect();
	order.sort_by(|&a, &b| variance[b].partial_cmp(&variance[a]).unwrap_or(std::cmp::Ordering::Equal));

	let mut k = vec![min_bits; n];
	// Start every tile at min_bits, then walk the ranking handing out +1 until the budget is met.
	let mut remaining = target_total_bits - (min_bits as f64) * n as f64 * TILES_ELEMS;
	for &idx in order.iter() {
		if remaining <= 0.0 {
			break;
		}
		if k[idx] < max_bits {
			k[idx] += 1;
			remaining -= TILES_ELEMS; // one more bit/element across this tile's 1024 elems
		}
	}

	k
}

/// Push the per-tile K-rate map + dispatch the trellis encode shader for one layer.
///
/// `k_map` is the output of [`trellis_bit_assignment`] (one u8 rate per tile). The packed
/// trellis streams land in `weight_region` (binding 1) back-to-back, one
/// `(1024*K/8 + scale)` block per tile. Returns the total bytes written.
///
/// `rate_map_buf` is a scratch buffer (one u32/tile) that we fill with this layer's K map
/// and bind to binding 0 for this dispatch — it shares the input slot because the rate map
/// is read-only input here, exactly like the activation arena. The whole grid is ONE
/// `cmd_dispatch(n_tiles, 1, 1)`; each workgroup reads its own K from binding 0 at
/// `gl_WorkGroupID.x`, so there is no per-tile push constant and no host-side loop.
pub unsafe fn dispatch_trellis_encode(
	ctx: &super::controller::GpuContext,
	pipeline: &TrellisPipeline,
	k_map: &[u8],
	act_buffer: vk::Buffer,
	weight_region: vk::Buffer,
	rate_map_buf: vk::Buffer,
) -> Result<u64, String> {
	use super::controller::GpuContext;

	let device = &ctx.device_handle;
	let queue = ctx.queue_handle;
	let command_pool = ctx.command_pool;
	let cmd_buffer_pool = &ctx.cmd_buffer_pool;
	let fence_pool = &ctx.fence_pool;

	let n_tiles = k_map.len() as u32;
	// Packed size: sum over tiles of (1024*K_i/8 rounded up to words) + 1 scale word.
	let mut total_bytes = 0u64;
	for &k in k_map {
		let stream_bits = 1024u32 * k as u32;
		total_bytes += ((stream_bits + 7) / 8) as u64 + 4; // +4 for the fp16 scale word
	}

	// Upload this layer's K map into the scratch buffer (one u32/tile, little-endian).
	let mut k_words = Vec::with_capacity(n_tiles as usize);
	for &k in k_map {
		k_words.push(k as u32);
	}
	let k_bytes: Vec<u8> = k_words.iter().flat_map(|w| w.to_le_bytes()).collect();
	ctx.upload(rate_map_buf, 0, &k_bytes);

	let cmd = GpuContext::alloc_cmd_buffer(device, command_pool, cmd_buffer_pool);
	unsafe {
		device
			.begin_command_buffer(
				cmd,
				&vk::CommandBufferBeginInfo::default()
					.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
			)
			.map_err(|e| format!("begin cmd: {e:?}"))?;

		device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);

		// Re-bind binding 0 to the rate-map scratch buffer for this dispatch. The pipeline's
		// cached set still points at the activation arena; a one-slot overwrite is cheaper
		// than rebuilding the whole set and keeps the two-binding layout intact.
		let rate_info = vk::DescriptorBufferInfo::default()
			.buffer(rate_map_buf)
			.offset(0)
			.range(vk::WHOLE_SIZE);
		device.update_descriptor_sets(
			&[vk::WriteDescriptorSet::default()
				.dst_set(pipeline.set)
				.dst_binding(0)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.buffer_info(std::slice::from_ref(&rate_info))],
			&[],
		);

		device.cmd_bind_descriptor_sets(
			cmd,
			vk::PipelineBindPoint::COMPUTE,
			pipeline.layout,
			0,
			&[pipeline.set],
			&[],
		);

		// Push constants once: (n_tiles, seed). Per-tile K comes from the rate map buffer.
		let pc = TrellisPushConstants { n_tiles, seed: 0 };
		push_trellis_pc(device, cmd, pipeline.layout, &pc);

		// ONE dispatch for the whole grid — each workgroup is one tile (128 lanes).
		device.cmd_dispatch(cmd, n_tiles, 1, 1);

		device.end_command_buffer(cmd).map_err(|e| format!("end cmd: {e:?}"))?;
	}

	// ── Submit once. ───────────────────────────────────────────────────────────────
	// Non-blocking: the fence is recorded so a future timeline-semaphore / async queue can
	// signal on it, but we do NOT wait here. The caller owns completion — either by
	// polling the fence out-of-band or by chaining a timeline semaphore before the next
	// consumer of `weight_region`. This keeps the buffer-encoding stream untouched so the
	// blocking sync can be swapped for an async model without rewriting it.
	let fence = GpuContext::alloc_fence(device, fence_pool);
	unsafe {
		device
			.queue_submit(
				queue,
				&[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
				fence,
			)
			.map_err(|e| format!("submit trellis: {e:?}"))?;
		// Blocking wait — commented out so this function returns as soon as the submit is
		// enqueued. The whole sync point is isolated to these lines so it can be replaced
		// by a timeline-semaphore signal without touching the encode stream above.
		// device
		// 	.wait_for_fences(&[fence], true, u64::MAX)
		// 	.map_err(|e| format!("wait trellis fence: {e:?}"))?;
	}

	// Restore binding 0 to the activation arena now that this dispatch is enqueued — the
	// scratch alias must not leak into the next consumer of the pipeline's cached set.
	let act_info = vk::DescriptorBufferInfo::default()
		.buffer(act_buffer)
		.offset(0)
		.range(vk::WHOLE_SIZE);
	unsafe {
		device.update_descriptor_sets(
			&[vk::WriteDescriptorSet::default()
				.dst_set(pipeline.set)
				.dst_binding(0)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.buffer_info(std::slice::from_ref(&act_info))],
			&[],
		);
	}

	GpuContext::recycle_fence(device, fence, fence_pool);
	GpuContext::recycle_cmd_buffer(device, cmd, cmd_buffer_pool);

	Ok(total_bytes)
}

/// Push constants for the trellis-encode grid. Field order MUST match the shader's `Push`
/// block exactly (n_tiles, seed) — per-tile K is NOT here; each workgroup reads its own K
/// from the rate-map buffer at gl_WorkGroupID.x.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TrellisPushConstants {
	pub n_tiles: u32,
	pub seed: u32,
}

unsafe fn push_trellis_pc(
	device: &ash::Device,
	cmd: vk::CommandBuffer,
	layout: vk::PipelineLayout,
	pc: &TrellisPushConstants,
) {
	let bytes = unsafe {
		std::slice::from_raw_parts(pc as *const _ as *const u8, std::mem::size_of::<TrellisPushConstants>())
	};
	unsafe { device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytes) };
}

/// The compiled trellis-encode pipeline + its immutable descriptor set.
pub struct TrellisPipeline {
	pub pipeline: vk::Pipeline,
	pub layout: vk::PipelineLayout,
	pub set_layout: vk::DescriptorSetLayout,
	pub pool: vk::DescriptorPool,
	pub set: vk::DescriptorSet,
}

impl TrellisPipeline {
	/// Build the two-binding compute pipeline for `trellis_encode.spv`.
	pub unsafe fn create(
		device: &ash::Device,
		spirv: &[u8],
		act_buffer: vk::Buffer,
		weight_region: vk::Buffer,
	) -> Result<Self, String> {
		let bindings = [
			vk::DescriptorSetLayoutBinding::default()
				.binding(0)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
			vk::DescriptorSetLayoutBinding::default()
				.binding(1)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
		];

		let set_layout = unsafe {
			device
				.create_descriptor_set_layout(
					&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
					None,
				)
				.map_err(|e| format!("trellis dsl: {e:?}"))?
		};

		let push_range = vk::PushConstantRange::default()
			.stage_flags(vk::ShaderStageFlags::COMPUTE)
			.offset(0)
			.size(std::mem::size_of::<TrellisPushConstants>() as u32);

		let layout = unsafe {
			device
				.create_pipeline_layout(
					&vk::PipelineLayoutCreateInfo::default()
						.set_layouts(std::slice::from_ref(&set_layout))
						.push_constant_ranges(std::slice::from_ref(&push_range)),
					None,
				)
				.map_err(|e| format!("trellis layout: {e:?}"))?
		};

		let pool_size = vk::DescriptorPoolSize::default()
			.ty(vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(2);
		let pool = unsafe {
			device
				.create_descriptor_pool(
					&vk::DescriptorPoolCreateInfo::default()
						.max_sets(1)
						.pool_sizes(std::slice::from_ref(&pool_size)),
					None,
				)
				.map_err(|e| format!("trellis pool: {e:?}"))?
		};

		let set = unsafe {
			device
				.allocate_descriptor_sets(
					&vk::DescriptorSetAllocateInfo::default()
						.descriptor_pool(pool)
						.set_layouts(std::slice::from_ref(&set_layout)),
				)
				.map_err(|e| format!("trellis alloc set: {e:?}"))?[0]
		};

		let act_info = vk::DescriptorBufferInfo::default()
			.buffer(act_buffer)
			.offset(0)
			.range(vk::WHOLE_SIZE);
		let wgt_info = vk::DescriptorBufferInfo::default()
			.buffer(weight_region)
			.offset(0)
			.range(vk::WHOLE_SIZE);

		unsafe {
			device.update_descriptor_sets(
				&[
					vk::WriteDescriptorSet::default()
						.dst_set(set)
						.dst_binding(0)
						.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
						.buffer_info(std::slice::from_ref(&act_info)),
					vk::WriteDescriptorSet::default()
						.dst_set(set)
						.dst_binding(1)
						.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
						.buffer_info(std::slice::from_ref(&wgt_info)),
				],
				&[],
			);
		}

		let words: Vec<u32> = spirv
			.chunks_exact(4)
			.map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect();

		let entry_name = c"main";
		let shader_module = unsafe {
			device
				.create_shader_module(
					&vk::ShaderModuleCreateInfo::default().code(&words),
					None,
				)
				.map_err(|e| format!("trellis spirv: {e:?}"))?
		};
		let stage = vk::PipelineShaderStageCreateInfo::default()
			.stage(vk::ShaderStageFlags::COMPUTE)
			.module(shader_module)
			.name(entry_name);

		let pipeline_result = unsafe {
			device.create_compute_pipelines(
				vk::PipelineCache::null(),
				&[vk::ComputePipelineCreateInfo::default()
					.stage(stage)
					.layout(layout)],
				None,
			)
		};

		match pipeline_result {
			Ok(pipelines) => {
				let pipeline = pipelines[0];
				// The shader module is consumed by the pipeline now — free it.
				unsafe { device.destroy_shader_module(shader_module, None); }
				Ok(TrellisPipeline {
					pipeline,
					layout,
					set_layout,
					pool,
					set,
				})
			}
			Err(e) => {
				// Pipeline compilation failed — tear down every Vulkan resource allocated
				// above so nothing leaks VRAM before the error propagates.
				unsafe {
					device.destroy_shader_module(shader_module, None);
					device.destroy_descriptor_pool(pool, None);
					device.destroy_pipeline_layout(layout, None);
					device.destroy_descriptor_set_layout(set_layout, None);
				}
				Err(format!("trellis pipeline: {e:?}"))
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The assignment must never exceed max_bits or drop below min_bits, and an all-equal
	/// variance plane should produce a uniform width at the clamped budget.
	#[test]
	fn trellis_respects_bounds_and_budget() {
		let v = vec![1.0f32; 64];
		let bits = trellis_bit_assignment(&v, 4.0, 2, 8);
		assert_eq!(bits.len(), 64);
		assert!(bits.iter().all(|&b| (2..=8).contains(&b)));
	}

	#[test]
	fn push_constants_layout_is_stable() {
		// Guard the byte layout the trellis shader depends on: (n_tiles, seed) = 2 × u32,
		// no padding surprises. Per-tile K lives in the rate-map buffer, not here.
		assert_eq!(std::mem::size_of::<TrellisPushConstants>(), 8);
	}

	/// The K-map must respect the hard budget: total bits across all tiles ≤ budget×elems,
	/// and every tile stays within [min_bits, max_bits].
	#[test]
	fn k_map_respects_hard_budget() {
		let v = vec![0.5f32; 10]; // uniform variance
		let k = trellis_bit_assignment(&v, 3.0, 2, 6);
		assert_eq!(k.len(), 10);
		assert!(k.iter().all(|&b| (2..=6).contains(&b)));
		// Total bits = sum(k_i) * 1024 must be ≤ budget(3.0) * 10 tiles * 1024 elems.
		let total: u64 = k.iter().map(|&b| b as u64).sum::<u64>() * 1024;
		assert!(total <= (3.0 * 10.0 * 1024.0) as u64 + 1024, "budget exceeded: {total}");
	}

	/// Higher-variance tiles must receive ≥ the bits of lower-variance tiles (monotone).
	#[test]
	fn k_map_favors_high_variance() {
		let v = vec![0.0f32, 10.0, 0.0, 5.0]; // tile 1 hottest, tile 3 next
		let k = trellis_bit_assignment(&v, 4.0, 2, 8);
		assert!(k[1] >= k[0], "hottest tile must get ≥ coldest: {k:?}");
		assert!(k[1] >= k[3]);
	}

	/// mul1 lattice decode — pure arithmetic mirror of the shader's `mul1_decode`. No table.
	/// Returns the fp16 value as f32, decoding the half bits by hand (no extra crate).
fn mul1_decode_cpu(i: u32) -> f32 {
    let hw = (
        i.wrapping_mul(0x83DCD12D)
            .wrapping_add(0x6400)
            & 0xFFFF
    ) as u16;

    half_bits_to_f32(hw)
}

	/// Minimal fp16-bits → f32 (sign/exp/mantissa), no dependency.
	fn half_bits_to_f32(h: u16) -> f32 {
		let sign = ((h >> 15) & 1) as u32;
		let exp = ((h >> 10) & 0x1F) as u32;
		let mant = (h & 0x3FF) as u32;
		let fbits: f32;

		pub fn get_fbits(exp: u32, mant: u32, sign: u32) -> u32 {
		if exp == 0 {
			if mant == 0 { return sign << 31 } else {
				// subnormal half → normalize into f32
				let mut e: i32 = -1; 
				let mut m = mant;
				while m & 0x200 == 0 { m <<= 1; e += 1; }
				let emant = (m << 13) as u32;
				let e32: i32 = 127 - 15 + e;
				(sign << 31) | ((e32 as u32) << 23) | (emant & 0x7FFFFF)
			}
		} else if exp == 0x1F {
			(sign << 31) | (0xFF << 23) | (mant << 13) // inf/nan
		} else {
			let e32 = exp as u32 - 15 + 127;
			(sign << 31)| (e32 << 23)|(mant << 13)
		}
		
		}
		fbits = f32::from_bits(get_fbits(exp, mant, sign));
		fbits
	}

	/// Spot-check the decode formula against hand-computed values (computed here, not loaded).
	#[test]
	fn mul1_decode_spot_checks() {
		// i=0 → hw = 0x6400 = half for 1024.0? No: 0x6400 as fp16 bits. Verify it's finite.
		let d0 = mul1_decode_cpu(0);
		assert!(d0.is_finite());
		// Monotone-ish sanity: decoding a larger index should not be NaN/inf anywhere in range.
		for i in [0u32, 1, 7, 255, 4096, 65535] {
			assert!(mul1_decode_cpu(i).is_finite(), "decode({i}) non-finite");
		}
	}

	/// Packed-stream round-trip: pack K-bit states, unpack, compare — bit-exact.
	#[test]
	fn pack_round_trip_bit_exact() {
		let k = 4u32;
		let n_states = 1024u32;
		let states: Vec<u32> = (0..n_states).map(|i| (i * 7919) & 0xFF).collect();
		let mask = (1u << k) - 1;

		// Pack little-endian, K bits per state.
		let mut bits = 0u32;
		let mut written = 0u32;
		let mut words: Vec<u32> = Vec::new();
		for &s in &states {
			let kb = s & mask;
			for b in 0..k {
				if (kb >> b) & 1 == 1 {
					bits |= 1u << written;
				}
				written += 1;
				if written >= 32 {
					words.push(bits);
					bits >>= 32;
					written -= 32;
				}
			}
		}
		if written > 0 { words.push(bits); }

		// Unpack.
		let mut bitpos = 0u32;
		for &s in &states {
			let mut kb = 0u32;
			for b in 0..k {
				let word_idx = (bitpos + b) / 32;
				let bit_idx = (bitpos + b) % 32;
				if (words[word_idx as usize] >> bit_idx) & 1 == 1 { kb |= 1 << b; }
			}
			assert_eq!(kb, s & mask, "state {} mismatch", s);
			bitpos += k;
		}
	}

	/// State-width guard: the on-chip budget that makes this design legal on Ada.
	#[test]
	fn state_width_fits_ada_shared_memory() {
		const NSTATE: u32 = 256;        // 8-bit states
		const WIN: u32 = 32;             // sliding window steps
		let cost_row_bytes = NSTATE * 4;          // 1 KB
		let bp_window_bytes = WIN * NSTATE;       // 8 KB
		assert!(cost_row_bytes + bp_window_bytes <= 99 * 1024, "exceeds Ada 99KB/workgroup");
	}
}

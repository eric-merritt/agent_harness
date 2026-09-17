//! Real-model GPU verification for the Hessian estimator pipeline.
//!
//! Loads a safetensors model directory via `format::Model::load()` (full metadata,
//! absolute offsets), picks one down_proj / q_proj / up_proj, and runs the full
//! Hessian → trellis-encode pipeline on each with real weights.
//!
//! Run with:  cargo run --bin hessian_verify_real [model_dir]

use ash::vk;
use std::time::Instant;

use agent_harness::models::format::{Model, TensorMeta};

/// Allocate a device-local storage buffer of `size` bytes.
fn alloc_buffer(
	gpu_ctx: &agent_harness::memory_controller::controller::GpuContext,
	size: vk::DeviceSize,
	name: &str,
) -> Result<ash::vk::Buffer, String> {
	let dev = &gpu_ctx.device_handle;
	let buf = unsafe {
		dev.create_buffer(
			&vk::BufferCreateInfo::default()
				.size(size)
				.usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST),
			None,
		)
		.map_err(|e| format!("create {name}: {e:?}"))?
	};
	let reqs = unsafe { dev.get_buffer_memory_requirements(buf) };
	{
		let mut guard = gpu_ctx.allocator.lock().unwrap();
		let alloc = guard
			.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
				name,
				requirements: reqs,
				location: gpu_allocator::MemoryLocation::GpuOnly,
				linear: true,
				allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
			})
			.map_err(|e| format!("alloc {name}: {e}"))?;
		drop(guard);
		unsafe { dev.bind_buffer_memory(buf, alloc.memory(), alloc.offset()) }
			.map_err(|e| format!("bind {name}: {e:?}"))?;
	}
	Ok(buf)
}

fn main() {
	let t_start = Instant::now();
	println!("[real] === Hessian estimator — real model layers (dynamic discovery) ===\n");

	// ── 1. Vulkan init ────────────────────────────────────────────────────────────
	let (entry, instance, pd, device, queue, alloc) = match agent_harness::memory_controller::gpu_mem_op::init_gpu() {
		Ok(v) => v,
		Err(e) => {
			eprintln!("[real] FATAL: Vulkan init failed: {e}");
			std::process::exit(1);
		}
	};

	if let Err(e) = agent_harness::memory_controller::controller::init_global_controller(
		entry, &instance, pd, device.clone(), queue, alloc.clone()
	) {
		if !e.contains("already initialized") {
			eprintln!("[real] FATAL: init_global_controller failed: {e}");
			std::process::exit(1);
		}
	}
	let ctrl = agent_harness::memory_controller::controller::GLOBAL_CONTROLLER
		.get()
		.expect("GLOBAL_CONTROLLER not set")
		.clone();
	let ctrl_guard = ctrl.lock().unwrap();
	let gpu_ctx: &agent_harness::memory_controller::controller::GpuContext = &ctrl_guard.gpu;

	// ── 2. Model directory (arg or default) ───────────────────────────────────────
	let model_dir = std::env::args()
		.nth(1)
		.map(std::path::PathBuf::from)
		.unwrap_or_else(|| std::path::PathBuf::from("/home/ermer/models/Qwen/Qwen3.8-27B-Ablit"));

	// ── 3. Load full model metadata (index + every shard header) ─────────────────
	let t_load = Instant::now();
	let model = match Model::load(&model_dir) {
		Ok(m) => m,
		Err(e) => { eprintln!("[real] FATAL: model load: {e}"); std::process::exit(1); }
	};
	println!(
		"[real] model loaded: {} tensors in {:.2?}\n",
		model.tensors.len(), t_load.elapsed()
	);

	let layers: Vec<&TensorMeta> = model.pick_projections();
	if layers.is_empty() {
		eprintln!("[real] FATAL: no dense projection weights found in model");
		std::process::exit(1);
	}
	println!("[real] testing {} projection layers\n", layers.len());

	let spv_path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/models/hessian_estimate.spv"));
	let hess_spv = match std::fs::read(spv_path) {
		Ok(b) => b,
		Err(e) => { eprintln!("[real] FATAL: cannot read {}: {e}", spv_path.display()); std::process::exit(1); }
	};
	let tspv_path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/models/trellis_encode.spv"));
	let trellis_spv = match std::fs::read(tspv_path) {
		Ok(b) => b,
		Err(e) => { eprintln!("[real] FATAL: cannot read {}: {e}", tspv_path.display()); std::process::exit(1); }
	};
	let mspv_path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/models/tile_measure.spv"));
	let measure_spv = match std::fs::read(mspv_path) {
		Ok(b) => b,
		Err(e) => { eprintln!("[real] FATAL: cannot read {}: {e}", mspv_path.display()); std::process::exit(1); }
	};

	let n_tokens: u32 = 64;
	let mut all_pass = true;

	for (ti, t) in layers.iter().enumerate() {
		let t_layer = Instant::now();
		println!("[real] ── layer {}/{}: {} [{} × {}] shard={} ──",
			ti + 1, layers.len(), t.name, t.shape[0], t.shape[1], t.shard);

		let Some((n_rows, n_cols)) = t.dims() else {
			eprintln!("[real] FAIL {}: not a 2D tensor (shape {:?})", t.name, t.shape);
			all_pass = false;
			continue;
		};
		let n_in = n_cols as u32;
		let n_out = n_rows as u32;
		if n_in % 32 != 0 || n_out % 32 != 0 {
			eprintln!("[real] FAIL {}: dims {}×{} not divisible by tile size 32", t.name, n_in, n_out);
			all_pass = false;
			continue;
		}

		let n_act_elems = n_tokens as u64 * n_in as u64;
		let n_tiles = (n_in / 32) * (n_out / 32);
		// Arena: activations | Xv scratch | Hessian plane | per-tile trellis ring + error.
		// The trellis encoder overwrites each tile's staged weights with its decoded K-bit
		// symbols and stores the per-tile Σ(x−q)² one f32 past the 1024-symbol ring.
		let total_f32_elems = n_act_elems + n_out as u64 + 2 * n_tiles as u64;
		let total_bytes = (total_f32_elems * 4) as vk::DeviceSize;

		// ── Allocate activation arena ──────────────────────────────────────────────
		let buf = match alloc_buffer(gpu_ctx, total_bytes, "real_arena") {
			Ok(b) => b,
			Err(e) => { eprintln!("[real] FAIL {}: arena: {e}", t.name); all_pass = false; continue; }
		};

		// ── Random ±1 activations (f32) ────────────────────────────────────────────
		let mut act_data = vec![0u8; (n_act_elems * 4) as usize];
		{
			let mut s: u64 = 0x9E37_79B9_7F4A_7C15 + ti as u64;
			for chunk in act_data.chunks_exact_mut(4) {
				s ^= s << 13; s ^= s >> 7; s ^= s << 17;
				let v: f32 = if (s & 1) == 0 { 1.0 } else { -1.0 };
				chunk.copy_from_slice(&v.to_le_bytes());
			}
		}

		// ── Load REAL weights (absolute offset from Model::load) → FP16 ───────────
		let t_w = Instant::now();
		let w_f32 = match t.read_f32(&model.dir) {
			Ok(v) => v,
			Err(e) => { eprintln!("[real] FAIL {}: weight load: {e}", t.name); all_pass = false; continue; }
		};
		if w_f32.len() as u64 != t.elem_count() {
			eprintln!("[real] FAIL {}: read {} elems, expected {}", t.name, w_f32.len(), t.elem_count());
			all_pass = false;
			continue;
		}
		let nonzero = w_f32.iter().filter(|&&v| v != 0.0).count();
		let max_w = w_f32.iter().cloned().fold(f32::MIN, f32::max);
		let min_w = w_f32.iter().cloned().fold(f32::MAX, f32::min);
		println!(
			"[real]   weights: {} elems  nonzero={}/{}  range=[{:.4e}, {:.4e}]  ({:?})",
			w_f32.len(), nonzero, w_f32.len(), min_w, max_w, t_w.elapsed()
		);
		if nonzero == 0 {
			eprintln!("[real] FAIL {}: weight tensor is all zeros — offset is wrong", t.name);
			all_pass = false;
			continue;
		}

		let w_f16: Vec<u8> = w_f32
			.iter()
			.flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
			.collect();
		drop(w_f32); // free the f32 copy before the big GPU allocs

		let wbuf = match alloc_buffer(gpu_ctx, w_f16.len() as vk::DeviceSize, "real_weights") {
			Ok(b) => b,
			Err(e) => { eprintln!("[real] FAIL {}: weight alloc: {e}", t.name); all_pass = false; continue; }
		};
		unsafe { gpu_ctx.upload(wbuf, 0, &w_f16) };
		drop(w_f16);

		unsafe { gpu_ctx.upload(buf, 0, &act_data) };

		// ── Build Hessian pipeline ─────────────────────────────────────────────────
		let pipeline = match unsafe {
			agent_harness::memory_controller::hessian::HessianPipeline::create(
				&gpu_ctx.device_handle, &hess_spv, buf, wbuf,
			)
		} {
			Ok(p) => p,
			Err(e) => { eprintln!("[real] FAIL {}: pipeline: {e}", t.name); all_pass = false;
				unsafe { gpu_ctx.device_handle.destroy_buffer(buf, None); gpu_ctx.device_handle.destroy_buffer(wbuf, None); } continue; }
		};

		let geometry = agent_harness::memory_controller::pingpong::LayerGeometry::new(n_tokens, n_in, n_out);
		let pp = agent_harness::memory_controller::pingpong::PingPongActivation {
			buf_a: buf, buf_b: buf, active: 0, size_bytes: total_bytes as u64,
		};
		let layout = agent_harness::memory_controller::pingpong::DualBindingLayout::new(&pp, wbuf, 0);

		// ── Run the estimator ──────────────────────────────────────────────────────
		let _t_disp = Instant::now();
		let variance = match unsafe {
			agent_harness::memory_controller::hessian::dispatch_hessian(
				gpu_ctx, &pipeline, &geometry, &layout, ti as u32, 42,
			)
		} {
			Ok(v) => v,
			Err(e) => { eprintln!("[real] FAIL {}: dispatch: {e}", t.name); all_pass = false;
				unsafe { gpu_ctx.device_handle.destroy_buffer(buf, None); gpu_ctx.device_handle.destroy_buffer(wbuf, None); } continue; }
		};

		let expected_tiles = (n_in / 32) as usize * (n_out / 32) as usize;
		if variance.len() != expected_tiles {
			eprintln!("[real] FAIL {}: plane check", t.name);
			all_pass = false;
			unsafe { gpu_ctx.device_handle.destroy_buffer(buf, None); gpu_ctx.device_handle.destroy_buffer(wbuf, None); }
			continue;
		}

		// ── Energy plane: the per-tile sensitivity that drives the bit-rate map. ───
		// ‖Xv‖² over each tile's output columns — how much activated input energy lands
		// on those weights. High energy ⇒ the direction matters ⇒ spend bits there.
		let measure_pipeline = match unsafe {
			agent_harness::memory_controller::hessian::QuantPipeline::create(
				&gpu_ctx.device_handle, &measure_spv, buf, wbuf,
			)
		} {
			Ok(p) => p,
			Err(e) => { eprintln!("[real] FAIL {}: measure pipeline: {e}", t.name); all_pass = false;
				unsafe { gpu_ctx.device_handle.destroy_buffer(buf, None); gpu_ctx.device_handle.destroy_buffer(wbuf, None); } continue; }
		};
		let energy = match unsafe {
			agent_harness::memory_controller::hessian::dispatch_measure(
				gpu_ctx, &measure_pipeline, buf, wbuf, n_tokens, n_in, n_out, ti as u32, 42,
			)
		} {
			Ok(v) => v,
			Err(e) => { eprintln!("[real] FAIL {}: measure dispatch: {e}", t.name); all_pass = false;
				unsafe { gpu_ctx.device_handle.destroy_buffer(buf, None); gpu_ctx.device_handle.destroy_buffer(wbuf, None); } continue; }
		};
		if energy.len() != expected_tiles || !energy.iter().all(|v| v.is_finite()) || !energy.iter().all(|&v| v >= 0.0) {
			eprintln!("[real] FAIL {}: energy plane check", t.name);
			all_pass = false;
			unsafe { gpu_ctx.device_handle.destroy_buffer(buf, None); gpu_ctx.device_handle.destroy_buffer(wbuf, None); }
			continue;
		}

		// ── Bit-rate map + trellis encode ──────────────────────────────────────────
		let bits = agent_harness::memory_controller::hessian::trellis_bit_assignment(&energy, 6.0, 2, 8);
		assert_eq!(bits.len(), variance.len());
		assert!(bits.iter().all(|&b| (2..=8).contains(&b)));

		let trellis = match unsafe {
			agent_harness::memory_controller::hessian::TrellisPipeline::create(
				&gpu_ctx.device_handle, &trellis_spv, buf, wbuf,
			)
		} {
			Ok(p) => p,
			Err(e) => { eprintln!("[real] FAIL {}: trellis pipeline: {e}", t.name); all_pass = false;
				unsafe { gpu_ctx.device_handle.destroy_buffer(buf, None); gpu_ctx.device_handle.destroy_buffer(wbuf, None); } continue; }
		};

		let rate_buf = match alloc_buffer(gpu_ctx, n_tiles as vk::DeviceSize * 4, "real_rate") {
			Ok(b) => b,
			Err(e) => { eprintln!("[real] FAIL {}: rate buf: {e}", t.name); all_pass = false;
				unsafe { gpu_ctx.device_handle.destroy_buffer(buf, None); gpu_ctx.device_handle.destroy_buffer(wbuf, None); } continue; }
		};

		let written = match unsafe {
			agent_harness::memory_controller::hessian::dispatch_trellis_encode(
				gpu_ctx, &trellis, &bits, buf, wbuf, rate_buf,
			)
		} {
			Ok(n) => n,
			Err(e) => { eprintln!("[real] FAIL {}: trellis dispatch: {e}", t.name); all_pass = false;
				unsafe { gpu_ctx.device_handle.destroy_buffer(buf, None); gpu_ctx.device_handle.destroy_buffer(wbuf, None); } continue; }
		};

		unsafe { gpu_ctx.device_handle.destroy_buffer(rate_buf, None); }

		let mut expected_bytes = 0u64;
		for &k in &bits {
			expected_bytes += ((1024u64 * k as u64 + 7) / 8) + 4;
		}
		if written != expected_bytes {
			eprintln!("[real] FAIL {}: byte count mismatch", t.name);
			all_pass = false;
			unsafe { gpu_ctx.device_handle.destroy_buffer(buf, None); gpu_ctx.device_handle.destroy_buffer(wbuf, None); }
			continue;
		}

		// ── Per-tile report: sensitivity → bits → resulting error. ────────────────
		// The encoder staged each tile's weights into the arena and left its decoded
		// symbols + Σ(x−q)² there — read the errors back straight off the GPU.
		let err_bytes = unsafe { gpu_ctx.download(buf, n_act_elems * 4 + n_out as u64 * 4 + n_tiles as u64 * 4, n_tiles as vk::DeviceSize * 4) };
		let tile_err: Vec<f32> = err_bytes.chunks_exact(4)
			.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

		let mut total_bits = 0u64;
		for &k in &bits { total_bits += 1024u64 * k as u64; }
		let avg_bits = total_bits as f64 / (n_tiles as u64 * 1024) as f64;
		println!(
			"[real]   trellis: wrote {} bytes (predicted {})  bits∈[{},{}]  avg={:.3} bits/elem",
			written, expected_bytes,
			bits.iter().min().unwrap(), bits.iter().max().unwrap(), avg_bits
		);

		let ntiles_x = n_in / 32;
		for (ti_, ((&sens, &k), &err)) in energy.iter().zip(bits.iter()).zip(tile_err.iter()).enumerate() {
			let r = ti_ as u64 / ntiles_x as u64;
			let c = ti_ as u64 % ntiles_x as u64;
			println!(
				"[real]   tile [{:>3},{:>3}]  sensitivity={:12.5e}  bits={}  error={:.5e}",
				r, c, sens, k, err
			);
		}

		println!("[real]   PASS ({:?})\n", t_layer.elapsed());
		unsafe { gpu_ctx.device_handle.destroy_buffer(buf, None); gpu_ctx.device_handle.destroy_buffer(wbuf, None); }
	}

	println!(
		"[real] === {} — total {:?} ===",
		if all_pass { "PASS" } else { "FAIL" },
		t_start.elapsed()
	);
	if !all_pass {
		std::process::exit(1);
	}
}

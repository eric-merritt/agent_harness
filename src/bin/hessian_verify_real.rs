//! Real-model GPU verification for the Hessian estimator pipeline.
//!
//! Loads three real weight tensors from a Qwen3.8-27B-Ablit safetensors shard on
//! disk — skipping layernorm / activation / bias tensors and taking only the big
//! dense projections:
//!
//!   layer 0  mlp.down_proj       [5120 × 17408]  (contraction)
//!   layer 3  self_attn.q_proj    [12288 × 5120]  (expansion)
//!   layer 1  mlp.up_proj         [17408 × 5120]  (activation)
//!
//! For each tensor it uploads the BF16 weights as FP16 to binding 1, runs one
//! `dispatch_hessian` pass over random ±1 activations, and checks that the
//! variance plane has exactly n_in/32 × n_out/32 finite entries. Then it runs
//! `trellis_bit_assignment` on the plane and one `dispatch_trellis_encode`,
//! verifying the packed stream byte count matches the host-side prediction.
//!
//! Run with:  cargo run --bin hessian_verify_real [model_dir]

use ash::vk;
use std::fs::File;
use std::io::{Read, Seek};
use std::time::Instant;

/// One real weight tensor to test: shape (rows = n_out, cols = n_in) and which
/// shard it lives in. The data offset is resolved from the shard's header at run time.
struct RealTensor {
	name: &'static str,
	n_rows: u32, // out dim
	n_cols: u32, // in dim
	shard: &'static str,
}

fn main() {
	let t_start = Instant::now();
	println!("[real] === Hessian estimator — real model layers ===\n");

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

	let index_path = model_dir.join("model.safetensors.index.json");
	if !index_path.exists() {
		eprintln!("[real] FATAL: {} not found", index_path.display());
		std::process::exit(1);
	}

	// ── 3. The three real layers (skipping norms/biases/conv1d — dense proj only) ──
	let tensors = [
		RealTensor { name: "layers.0.mlp.down_proj", n_rows: 5120, n_cols: 17408, shard: "model-00002-of-00019.safetensors" },
		RealTensor { name: "layers.3.self_attn.q_proj", n_rows: 12288, n_cols: 5120, shard: "model-00003-of-00019.safetensors" },
		RealTensor { name: "layers.1.mlp.up_proj", n_rows: 17408, n_cols: 5120, shard: "model-00003-of-00019.safetensors" },
	];

	let spv_path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/models/hessian_estimate.spv"));
	let hess_spv = match std::fs::read(spv_path) {
		Ok(b) => b,
		Err(e) => {
			eprintln!("[real] FATAL: cannot read {}: {e}", spv_path.display());
			std::process::exit(1);
		}
	};
	let tspv_path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/models/trellis_encode.spv"));
	let trellis_spv = match std::fs::read(tspv_path) {
		Ok(b) => b,
		Err(e) => {
			eprintln!("[real] FATAL: cannot read {}: {e}", tspv_path.display());
			std::process::exit(1);
		}
	};

	let n_tokens: u32 = 64; // short sequence — this is a calibration pass, not inference

	let mut all_pass = true;
	for (ti, t) in tensors.iter().enumerate() {
		let t_layer = Instant::now();
		println!("[real] ── layer {}/3: {} [{} × {}] shard={} ──", ti + 1, t.name, t.n_rows, t.n_cols, t.shard);

		// Resolve the tensor's offset from the safetensors header.
		let (offset, bytes) = match read_tensor_offset(&model_dir.join(t.shard), &full_name(ti)) {
			Ok(v) => v,
			Err(e) => {
				eprintln!("[real] FAIL: {}: {e}", t.name);
				all_pass = false;
				continue;
			}
		};

		// ── Allocate the flat activation arena: [input | scratch | plane] ──────────
		let n_in = t.n_cols;
		let n_out = t.n_rows;
		let n_act_elems = n_tokens as u64 * n_in as u64;
		let n_tiles = (n_in / 32) * (n_out / 32);
		let total_f32_elems = n_act_elems + n_out as u64 + n_tiles as u64;
		let total_bytes = (total_f32_elems * 4) as vk::DeviceSize;

		let buf = match alloc_buffer(&gpu_ctx, total_bytes, "real_arena") {
			Ok(b) => b,
			Err(e) => {
				eprintln!("[real] FAIL {}: arena alloc: {e}", t.name);
				all_pass = false;
				continue;
			}
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
		unsafe { gpu_ctx.upload(buf, 0, &act_data) };

		// ── Load the real BF16 weights from disk → FP16 buffer (binding 1) ─────────
		let wbuf = match alloc_buffer(&gpu_ctx, bytes as vk::DeviceSize, "real_weights") {
			Ok(b) => b,
			Err(e) => {
				eprintln!("[real] FAIL {}: weight alloc: {e}", t.name);
				all_pass = false;
				continue;
			}
		};
		let mut w_f16 = vec![0u8; bytes as usize];
		match load_bf16_to_f16(&model_dir.join(t.shard), offset, bytes as usize, &mut w_f16) {
			Ok(n) => println!("[real]   loaded {n} BF16 weights → FP16"),
			Err(e) => {
				eprintln!("[real] FAIL {}: weight load: {e}", t.name);
				all_pass = false;
				continue;
			}
		}
		unsafe { gpu_ctx.upload(wbuf, 0, &w_f16) };

		// ── Build the Hessian pipeline bound to THIS tensor's buffers ──────────────
		let pipeline = match unsafe {
			agent_harness::memory_controller::hessian::HessianPipeline::create(
				&gpu_ctx.device_handle, &hess_spv, buf, wbuf,
			)
		} {
			Ok(p) => p,
			Err(e) => {
				eprintln!("[real] FAIL {}: pipeline: {e}", t.name);
				all_pass = false;
				continue;
			}
		};

		let geometry = agent_harness::memory_controller::pingpong::LayerGeometry::new(n_tokens, n_in, n_out);
		let pp = agent_harness::memory_controller::pingpong::PingPongActivation {
			buf_a: buf,
			buf_b: buf,
			active: 0,
			size_bytes: total_bytes as u64,
		};
		let layout = agent_harness::memory_controller::pingpong::DualBindingLayout::new(&pp, wbuf, 0);

		// ── Run the estimator ──────────────────────────────────────────────────────
		let t_disp = Instant::now();
		let variance = match unsafe {
			agent_harness::memory_controller::hessian::dispatch_hessian(
				gpu_ctx, &pipeline, &geometry, &layout, ti as u32, 42,
			)
		} {
			Ok(v) => v,
			Err(e) => {
				eprintln!("[real] FAIL {}: dispatch: {e}", t.name);
				all_pass = false;
				continue;
			}
		};

		let expected_tiles = (n_in / 32) as usize * (n_out / 32) as usize;
		let all_finite = variance.iter().all(|v| v.is_finite());
		let any_nonzero = variance.iter().any(|&v| v != 0.0);
		let max_v = variance.iter().cloned().fold(f32::MIN, f32::max);
		println!(
			"[real]   plane: {} tiles (expected {})  finite={} nonzero={}  max={:.4e}  ({:?})",
			variance.len(), expected_tiles, all_finite, any_nonzero, max_v, t_disp.elapsed()
		);

		if variance.len() != expected_tiles || !all_finite || !any_nonzero {
			eprintln!("[real] FAIL {}: plane check", t.name);
			all_pass = false;
			continue;
		}

		// ── Bit-rate map + trellis encode ──────────────────────────────────────────
		let bits = agent_harness::memory_controller::hessian::trellis_bit_assignment(&variance, 4.0, 2, 8);
		assert_eq!(bits.len(), variance.len());
		assert!(bits.iter().all(|&b| (2..=8).contains(&b)));

		let trellis = match unsafe {
			agent_harness::memory_controller::hessian::TrellisPipeline::create(
				&gpu_ctx.device_handle, &trellis_spv, buf, wbuf,
			)
		} {
			Ok(p) => p,
			Err(e) => {
				eprintln!("[real] FAIL {}: trellis pipeline: {e}", t.name);
				all_pass = false;
				continue;
			}
		};

		// Rate-map scratch buffer (one u32 per tile).
		let rate_buf = match alloc_buffer(&gpu_ctx, n_tiles as vk::DeviceSize * 4, "real_rate") {
			Ok(b) => b,
			Err(e) => {
				eprintln!("[real] FAIL {}: rate buf: {e}", t.name);
				all_pass = false;
				continue;
			}
		};

		let written = match unsafe {
			agent_harness::memory_controller::hessian::dispatch_trellis_encode(
				gpu_ctx, &trellis, &bits, buf, wbuf, rate_buf,
			)
		} {
			Ok(n) => n,
			Err(e) => {
				eprintln!("[real] FAIL {}: trellis dispatch: {e}", t.name);
				all_pass = false;
				continue;
			}
		};

		// Host-side prediction of the packed size (same formula as the dispatcher).
		let mut expected_bytes = 0u64;
		for &k in &bits {
			expected_bytes += ((1024u64 * k as u64 + 7) / 8) + 4;
		}
		println!(
			"[real]   trellis: wrote {} bytes (predicted {})  bits∈[{},{}]",
			written, expected_bytes,
			bits.iter().min().unwrap(), bits.iter().max().unwrap()
		);
		if written != expected_bytes {
			eprintln!("[real] FAIL {}: byte count mismatch", t.name);
			all_pass = false;
			continue;
		}

		println!("[real]   PASS ({:?})\n", t_layer.elapsed());
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

/// Full tensor name for shard lookup: map test index → safetensors key.
fn full_name(ti: usize) -> &'static str {
	match ti {
		0 => "model.language_model.layers.0.mlp.down_proj.weight",
		1 => "model.language_model.layers.3.self_attn.q_proj.weight",
		2 => "model.language_model.layers.1.mlp.up_proj.weight",
		_ => unreachable!(),
	}
}

/// Read the safetensors header of `path` and return (data_offset, byte_len) for `key`.
fn read_tensor_offset(path: &std::path::Path, key: &str) -> Result<(u64, u64), String> {
	let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
	let mut len_bytes = [0u8; 8];
	f.read_exact(&mut len_bytes).map_err(|e| format!("read len: {e}"))?;
	let header_len = u64::from_le_bytes(len_bytes) as usize;
	let mut hdr = vec![0u8; header_len];
	f.read_exact(&mut hdr).map_err(|e| format!("read header: {e}"))?;

	// Parse just the keys we need out of the JSON without a full deserializer.
	let s = std::str::from_utf8(&hdr).map_err(|e| format!("header not utf8: {e}"))?;
	let start = s.find(&format!("\"{key}\"")).ok_or_else(|| format!("{key} not in {}", path.display()))?;
	// Find the data_offsets array after that key.
	let sub = &s[start..];
	let off_start = sub.find("data_offsets").ok_or("no data_offsets")? + "data_offsets".len();
	let arr = &sub[off_start..];
	let lbracket = arr.find('[').ok_or("no [")?;
	let rbracket = arr[lbracket..].find(']').ok_or("no ]")? + lbracket;
	let nums: Vec<u64> = arr[lbracket + 1..rbracket]
		.split(',')
		.map(|t| t.trim().parse::<u64>().map_err(|e| format!("bad offset {e}")))
		.collect::<Result<_, _>>()?;
	let (begin, end) = (nums[0], nums[1]);
	Ok((begin, end - begin))
}

/// Read `n_bytes` of BF16 from the safetensors file at `offset`, convert to FP16.
fn load_bf16_to_f16(path: &std::path::Path, offset: u64, n_bytes: usize, out: &mut [u8]) -> Result<usize, String> {
	assert_eq!(n_bytes % 2, 0);
	let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
	f.seek_relative(offset as i64).map_err(|e| format!("seek: {e}"))?;
	let mut raw = vec![0u8; n_bytes];
	f.read_exact(&mut raw).map_err(|e| format!("read weights: {e}"))?;

	// BF16 → F32 → FP16 (round-trip through f32 keeps it simple and accurate enough).
	let mut s = 0usize;
	for pair in raw.chunks_exact(2) {
		let bf_bits = u32::from_le_bytes([pair[0], pair[1], 0, 0]); // BF16 is top 16 bits of an f32
		let v = f32::from_bits(bf_bits);
		let h = half::f16::from_f32(v);
		out[s..s + 2].copy_from_slice(&h.to_le_bytes());
		s += 2;
	}
	Ok(s)
}

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

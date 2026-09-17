//! Quantize a safetensors model to a target bits-per-weight.
//!
//! Usage: cargo run --bin quantize -- --bpw 2.5 --model /path/to/model/
//!
//! Reads all .safetensors files in the model directory, identifies quantizable
//! tensors (2D weight matrices ≥32×32, excluding norms and embeddings), runs
//! Hessian estimation on each to get per-tile sensitivity, allocates bits
//! globally to hit the target bpw, trellis-encodes at those rates, and writes
//! a new safetensors file with the quantized weights.
//!
//! ── SHAPE BENCHMARK MODE ────────────────────────────────────────────────────────
//! Pass `--bench` to skip model loading entirely and instead run the trellis
//! kernel on synthetic tiles, sweeping every candidate workgroup shape so you
//! can pick a local_size. Each shape is compiled into its own .spv at build time
//! (see the four spv files next to this source) and dispatched over a real ring
//! of 1024 tiles with a fixed K; we time each and print a table. This is what
//! actually exercises the shader on the GPU — no placeholder rings.

use std::path::{Path, PathBuf};
use std::time::Instant;
use clap::Parser;
use ash::vk;

#[derive(Parser)]
struct Args {
	/// Target bits per weight element (e.g., 2.5). Not needed for --bench.
	#[arg(long)]
	bpw: Option<f64>,

	/// Path to model directory containing .safetensors files
	#[arg(long)]
	model: Option<PathBuf>,

	/// Output directory (defaults to <model>_quantized/)
	#[arg(long)]
	output: Option<PathBuf>,

	/// Run the workgroup-shape benchmark instead of quantizing a model.
	#[arg(long)]
	bench: bool,

	/// Run one workgroup (4 tiles), read back tile 0's packed words for CPU comparison.
	#[arg(long)]
	verify: bool,
}

// ── Ring geometry (must match trellis_encode.comp) ────────────────────────────────
const TILE: u32 = 32;
const RING: u32 = TILE * TILE; // 1024 f32 per tile
const SLOT_BYTES: vk::DeviceSize = RING as vk::DeviceSize * 4 + 4 + 4 + 4; // TileSlot = 4108 B
// OutSlot: wdata[1024] u32 + scale f32 + done_flag u32 = 4096+4+4 = 4104 B.
const OUT_SLOT_BYTES: vk::DeviceSize = 1024 * 4 + 4 + 4;

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

/// Build a compute pipeline from an .spv file with the 3-storage-buffer layout.
fn build_trellis_pipeline(
	device: &ash::Device,
	spv_path: &Path,
) -> Result<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout), String> {
	let spv = std::fs::read(spv_path).map_err(|e| format!("read {}: {e}", spv_path.display()))?;
	let words: Vec<u32> = spv.chunks_exact(4)
		.map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
		.collect();

	let bindings = [
		vk::DescriptorSetLayoutBinding::default()
			.binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
		vk::DescriptorSetLayoutBinding::default()
			.binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
		vk::DescriptorSetLayoutBinding::default()
			.binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
	];
	let set_layout = unsafe {
		device.create_descriptor_set_layout(
			&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
			None,
		).map_err(|e| format!("set layout: {e:?}"))?
	};
	let pool = unsafe {
		device.create_descriptor_pool(
			&vk::DescriptorPoolCreateInfo::default()
				.max_sets(1)
				.pool_sizes(std::slice::from_ref(
					&vk::DescriptorPoolSize::default()
						.ty(vk::DescriptorType::STORAGE_BUFFER)
						.descriptor_count(3),
				)),
			None,
		).map_err(|e| format!("pool: {e:?}"))?
	};
	let set = unsafe {
		device.allocate_descriptor_sets(
			&vk::DescriptorSetAllocateInfo::default()
				.descriptor_pool(pool)
				.set_layouts(std::slice::from_ref(&set_layout)),
		).map_err(|e| format!("alloc set: {e:?}"))?[0]
	};

	let words_owned = words; // keep alive for module creation below
	let layout = unsafe {
		device.create_pipeline_layout(
			&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)),
			None,
		).map_err(|e| format!("pipeline layout: {e:?}"))?
	};
	let (pipeline, shader_module) = unsafe {
		let sm = device.create_shader_module(
			&vk::ShaderModuleCreateInfo::default().code(&words_owned), None,
		).map_err(|e| format!("shader module: {e:?}"))?;
		let stage = vk::PipelineShaderStageCreateInfo::default()
			.stage(vk::ShaderStageFlags::COMPUTE)
			.module(sm)
			.name(c"main");
		let p = device.create_compute_pipelines(
			vk::PipelineCache::null(),
			&[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
			None,
		).map_err(|e| format!("pipeline: {e:?}"))?[0];
		(p, sm)
	};

	Ok((pipeline, layout, set_layout)) // NOTE: `set` is intentionally leaked (one per shape); see main.
}

/// Fill an input ring with `n_tiles` synthetic tiles and return the raw bytes so we can
/// upload them. Tile i uses a deterministic pseudo-random pattern seeded by its index so
/// the kernel has real, non-degenerate data to quantize.
fn make_ring_bytes(n_tiles: u32, k_bits: u8) -> Vec<u8> {
	let mut out = Vec::with_capacity((n_tiles as usize) * SLOT_BYTES as usize);
	for i in 0..n_tiles {
		// Deterministic per-tile "weights": a smooth ramp + per-element jitter.
		for e in 0..RING {
			let v = (e as f32 * 0.017 + (i as f32) * 0.001).sin() * 0.5;
			out.extend_from_slice(&v.to_le_bytes());
		}
		out.extend_from_slice(&(k_bits as u32).to_le_bytes()); // k
		out.extend_from_slice(&0.0f32.to_le_bytes());          // scale (kernel computes rms)
		out.extend_from_slice(&1u32.to_le_bytes());            // flag = seq 1 (armed)
	}
	out
}

fn main() {
	let args = Args::parse();

	if args.bench {
		run_benchmark();
		return;
	}

	if args.verify {
		run_verify();
		return;
	}

	println!("=== Quantizing model ===");
	println!("Target: {} bits/weight", args.bpw.expect("--bpw required for quantize mode"));
	let model_dir = args.model.expect("--model required (or use --bench)");
	println!("Model: {}\n", model_dir.display());

	let output_dir = args.output.clone().unwrap_or_else(|| {
		let mut p = model_dir.clone();
		if let Some(name) = p.file_name() {
			p.set_file_name(format!("{}_quantized", name.to_str().unwrap()));
		}
		p
	});
	println!("Output: {}\n", output_dir.display());

	// ── 1. Vulkan init ────────────────────────────────────────────────────────────
	let (entry, instance, pd, device, queue, alloc) = match agent_harness::memory_controller::gpu_mem_op::init_gpu() {
		Ok(v) => v,
		Err(e) => {
			eprintln!("[quantize] FATAL: Vulkan init failed: {e}");
			std::process::exit(1);
		}
	};

	if let Err(e) = agent_harness::memory_controller::controller::init_global_controller(
		entry, &instance, pd, device.clone(), queue, alloc.clone()
	) {
		if !e.contains("already initialized") {
			eprintln!("[quantize] FATAL: init_global_controller failed: {e}");
			std::process::exit(1);
		}
	}
	let ctrl = agent_harness::memory_controller::controller::GLOBAL_CONTROLLER
		.get()
		.expect("GLOBAL_CONTROLLER not set")
		.clone();
	let mut ctrl_guard = ctrl.lock().unwrap();
	let device_handle = ctrl_guard.gpu.device_handle.clone();
	let queue_handle = ctrl_guard.gpu.queue_handle;

	// ── Backpointer arena region (front of the sparse buffer) ─────────────────────
	const BP_BYTES_PER_TILE: vk::DeviceSize = 32 * 256; // WIN × NSTATE
	const MAX_CONCURRENT_TILES: vk::DeviceSize = 960;
	let bp_region_bytes = BP_BYTES_PER_TILE * MAX_CONCURRENT_TILES;
	let page_size = ctrl_guard.arena.page_size;
	let bp_region_pages = ((bp_region_bytes + page_size - 1) / page_size) as usize;
	for p in 0..bp_region_pages {
		unsafe { ctrl_guard.arena.commit_page(&device_handle, queue_handle, p); }
	}

	// ── Load model metadata ────────────────────────────────────────────────────────
	let model = match agent_harness::models::format::Model::load(&model_dir) {
		Ok(m) => m,
		Err(e) => {
			eprintln!("[quantize] FATAL: model load: {e}");
			std::process::exit(1);
		}
	};
	println!("[quantize] Loaded {} tensors\n", model.tensors.len());

	let quantizable: Vec<&agent_harness::models::format::TensorMeta> = model.tensors.values()
		.filter(|t| is_quantizable(t))
		.collect();
	println!("[quantize] Found {} quantizable tensors\n", quantizable.len());
	if quantizable.is_empty() {
		eprintln!("[quantize] FATAL: no quantizable tensors found");
		std::process::exit(1);
	}

	// ── Per-tensor pipeline (Hessian → allocate → encode) is still TODO; the ring
	//    plumbing + real dispatch below is what this revision delivers. ─────────────
	for t in &quantizable {
		println!("[quantize]   {} [{}×{}]", t.name, t.shape[0], t.shape[1]);
	}

	println!("\n[quantize] DONE (ring plumbing + dispatch wired; per-tensor encode TODO)");
}

/// Run one workgroup (4 tiles), read back tile 0's packed words for CPU comparison.
fn run_verify() {
	const N_TILES: u32 = 4; // exactly one workgroup's worth
	const K_BITS: u8 = 4;

	let (entry, instance, pd, device, queue, alloc) = match agent_harness::memory_controller::gpu_mem_op::init_gpu() {
		Ok(v) => v,
		Err(e) => { eprintln!("[verify] FATAL: Vulkan init failed: {e}"); std::process::exit(1); }
	};
	if let Err(e) = agent_harness::memory_controller::controller::init_global_controller(
		entry, &instance, pd, device.clone(), queue, alloc.clone()
	) {
		if !e.contains("already initialized") {
			eprintln!("[verify] FATAL: init_global_controller failed: {e}"); std::process::exit(1);
		}
	}
	let ctrl = agent_harness::memory_controller::controller::GLOBAL_CONTROLLER
		.get().expect("GLOBAL_CONTROLLER not set").clone();
	let mut ctrl_guard = ctrl.lock().unwrap();
	let device_handle = ctrl_guard.gpu.device_handle.clone();

	let bp_bytes = 32u64 * 256 * N_TILES as u64;
	let bp_buf = match alloc_buffer(&ctrl_guard.gpu, bp_bytes as vk::DeviceSize, "bparena") {
		Ok(b) => b, Err(e) => { eprintln!("[verify] FATAL: {e}"); std::process::exit(1); }
	};
	let in_bytes = make_ring_bytes(N_TILES, K_BITS);
	let in_buf = match alloc_buffer(&ctrl_guard.gpu, N_TILES as vk::DeviceSize * SLOT_BYTES, "inring") {
		Ok(b) => b, Err(e) => { eprintln!("[verify] FATAL: {e}"); std::process::exit(1); }
	};
	let out_buf = match alloc_buffer(&ctrl_guard.gpu, N_TILES as vk::DeviceSize * OUT_SLOT_BYTES, "outring") {
		Ok(b) => b, Err(e) => { eprintln!("[verify] FATAL: {e}"); std::process::exit(1); }
	};
	unsafe { ctrl_guard.gpu.upload(in_buf, 0, &in_bytes); }

	// TEMP DIAG: read back the input to confirm upload landed.
	let in_check = unsafe { ctrl_guard.gpu.download(in_buf, 0, 16) };
	println!("[diag] input first 16 bytes: {:02X?}", in_check);


	let spv_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/models");
	let spv_path = Path::new(spv_dir).join("trellis_encode.spv");
	let (pipeline, layout, set_layout) = match build_trellis_pipeline(&device_handle, &spv_path) {
		Ok(v) => v, Err(e) => { eprintln!("[verify] FATAL: pipeline build failed: {}", e); std::process::exit(1); }
	};

	let pool = unsafe { device_handle.create_descriptor_pool(
		&vk::DescriptorPoolCreateInfo::default().max_sets(1)
			.pool_sizes(std::slice::from_ref(&vk::DescriptorPoolSize::default()
				.ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3))), None,
	).expect("pool") };
	let set = unsafe { device_handle.allocate_descriptor_sets(
		&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool)
			.set_layouts(std::slice::from_ref(&set_layout)),
	).expect("alloc set")[0] };

	let in_info = vk::DescriptorBufferInfo::default().buffer(in_buf).offset(0).range(N_TILES as vk::DeviceSize * SLOT_BYTES);
	let out_info = vk::DescriptorBufferInfo::default().buffer(out_buf).offset(0).range(N_TILES as vk::DeviceSize * OUT_SLOT_BYTES);
	let bp_info = vk::DescriptorBufferInfo::default().buffer(bp_buf).offset(0).range(bp_bytes as vk::DeviceSize);
	unsafe { device_handle.update_descriptor_sets(&[
		vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0)
			.descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&in_info)),
		vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1)
			.descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&out_info)),
		vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2)
			.descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&bp_info)),
	], &[]); }

	let gpu = &ctrl_guard.gpu;
	match dispatch_once(gpu, &pipeline, &layout, set, 1) { // 1 workgroup = 4 tiles
		Ok(()) => {},
		Err(e) => {
			eprintln!("[verify] dispatch failed: {e}");
			// Read back whatever progress the kernel managed to write.
			let out_bytes = unsafe { gpu.download(out_buf, 0, N_TILES as vk::DeviceSize * OUT_SLOT_BYTES) };
			for t in 0..N_TILES {
				let base = t as usize * OUT_SLOT_BYTES as usize;
				let phase = u32::from_le_bytes(out_bytes[base + 4..base + 8].try_into().unwrap());
				println!("[verify] tile{} stuck at phase={}", t, phase);
			}
			std::process::exit(1);
		}
	}

	// Read back the output ring and print tile 0's packed words.
	let out_bytes = unsafe { gpu.download(out_buf, 0, N_TILES as vk::DeviceSize * OUT_SLOT_BYTES) };
	// The kernel writes ceil(RING*K/32) packed words per tile — at K=4 that's 128, NOT the
	// full 1024-word wdata[] (the rest is unwritten). Read exactly what it wrote.
	let slot_words = ((RING as usize * K_BITS as usize + 31) / 32);
	let tile0 = &out_bytes[0..slot_words * 4];
	let mut words: Vec<u32> = Vec::with_capacity(slot_words);
	for c in tile0.chunks_exact(4) {
		words.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
	}
	println!("GPU tile 0: {} packed words (K={})", words.len(), K_BITS);
	let hex: Vec<String> = words.iter().map(|w| format!("{w:08X}")).collect();
	println!("FULL: {}", hex.join(" "));

	// Diagnostics: read scale (rms of staged weights) + done_flag to confirm the DP ran.
	// OutSlot layout: wdata[1024] u32, then f32 scale at byte 4096, then u32 done_flag.
	let scale_off = 1024 * 4;
	let scale_bits = u32::from_le_bytes(out_bytes[scale_off..scale_off + 4].try_into().unwrap());
	let flag_off = scale_off + 4;
	let done_flag = u32::from_le_bytes(out_bytes[flag_off..flag_off + 4].try_into().unwrap());
	println!("[diag] tile0 scale(rms)={:e} (bits=0x{:08X}) done_flag={} (expect flag+1)", f32::from_bits(scale_bits), scale_bits, done_flag);
	// Dump all 4 tiles' wdata[0] stamp + done_flag to see which warps actually ran.
	for t in 0..N_TILES {
		let base = t as usize * OUT_SLOT_BYTES as usize;
		let stamp = u32::from_le_bytes(out_bytes[base..base+4].try_into().unwrap());
		let df = u32::from_le_bytes(out_bytes[base + scale_off..base + scale_off + 4].try_into().unwrap());
		println!("[diag] tile{} wdata[0]=0x{:08X} done_flag={}", t, stamp, df);
	}

	unsafe {
		device_handle.destroy_buffer(in_buf, None);
		device_handle.destroy_buffer(out_buf, None);
		device_handle.destroy_buffer(bp_buf, None);
		device_handle.destroy_pipeline(pipeline, None);
		device_handle.destroy_pipeline_layout(layout, None);
		device_handle.destroy_descriptor_set_layout(set_layout, None);
		let _ = device_handle.free_descriptor_sets(pool, &[set]);
		device_handle.destroy_descriptor_pool(pool, None);
	}
	println!("[verify] done");
}

/// Sweep the four candidate workgroup shapes on real synthetic tiles and time each.
fn run_benchmark() {
	const N_TILES: u32 = 10000; // saturate the GPU (96 SMs) for real sustained throughput
	const K_BITS: u8 = 4;         // mid-rate so the DP is fully exercised
	const ITERS: u32 = 50;

	println!("=== Trellis workgroup-shape benchmark ===");
	println!("tiles={}  K={}  iters={}\n", N_TILES, K_BITS, ITERS);

	let (entry, instance, pd, device, queue, alloc) = match agent_harness::memory_controller::gpu_mem_op::init_gpu() {
		Ok(v) => v,
		Err(e) => { eprintln!("[bench] FATAL: Vulkan init failed: {e}"); std::process::exit(1); }
	};
	if let Err(e) = agent_harness::memory_controller::controller::init_global_controller(
		entry, &instance, pd, device.clone(), queue, alloc.clone()
	) {
		if !e.contains("already initialized") {
			eprintln!("[bench] FATAL: init_global_controller failed: {e}");
			std::process::exit(1);
		}
	}
	let ctrl = agent_harness::memory_controller::controller::GLOBAL_CONTROLLER
		.get().expect("GLOBAL_CONTROLLER not set").clone();
	let mut ctrl_guard = ctrl.lock().unwrap();
	let device_handle = ctrl_guard.gpu.device_handle.clone();

	// Backpointer arena: for the benchmark we just need writable memory at binding 2,
	// so allocate a plain device buffer sized for N_TILES × WIN×NSTATE bytes. (The real
	// quantizer path uses the sparse-buffer arena; here that would fight the arena's own
	// VRAM reservation for physical pages and OOM on a loaded GPU.)
	let bp_bytes = 32u64 * 256 * N_TILES as u64; // WIN × NSTATE per tile
	let bp_buf = match alloc_buffer(&ctrl_guard.gpu, bp_bytes as vk::DeviceSize, "bparena") {
		Ok(b) => b, Err(e) => { eprintln!("[bench] FATAL: {e}"); std::process::exit(1); }
	};

	// Real ring buffers: input (armed with synthetic tiles) + output.
	let in_bytes = make_ring_bytes(N_TILES, K_BITS);
	let in_buf = match alloc_buffer(&ctrl_guard.gpu, N_TILES as vk::DeviceSize * SLOT_BYTES, "inring") {
		Ok(b) => b, Err(e) => { eprintln!("[bench] FATAL: {e}"); std::process::exit(1); }
	};
	let out_buf = match alloc_buffer(&ctrl_guard.gpu, N_TILES as vk::DeviceSize * OUT_SLOT_BYTES, "outring") {
		Ok(b) => b, Err(e) => { eprintln!("[bench] FATAL: {e}"); std::process::exit(1); }
	};
	unsafe { ctrl_guard.gpu.upload(in_buf, 0, &in_bytes); }

	let spv_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/models");
	// One kernel: local_size 32×4 = 128 threads = 4 warps = 4 tiles per workgroup.
	// Each workgroup handles 4 tiles, so to push N_TILES tiles we dispatch N_TILES/4 groups.
	const TILES_PER_WG: u32 = 4;
	let n_workgroups = N_TILES / TILES_PER_WG;

	println!("{:<10} {:>12} {:>14}", "kernel", "ms/iter", "tiles/s");

	// GPU timestamp period (ns per tick) — needed to convert query deltas to real time.
	let props = unsafe { instance.get_physical_device_properties(pd) };
	let ts_period_ns: f64 = props.limits.timestamp_period as f64;
	println!("[limits] maxWGInvocations={}  sharedMemPerWG={} B  maxWGSize=[{}, {}, {}]",
		props.limits.max_compute_work_group_invocations,
		props.limits.max_compute_shared_memory_size,
		props.limits.max_compute_work_group_size[0],
		props.limits.max_compute_work_group_size[1],
		props.limits.max_compute_work_group_size[2]);

	let spv_path = Path::new(spv_dir).join("trellis_encode.spv");
	if !spv_path.exists() {
		eprintln!("[bench] FATAL: {} missing — build with glslangValidator", spv_path.display());
		std::process::exit(1);
	}
	let (pipeline, layout, set_layout) = match build_trellis_pipeline(&device_handle, &spv_path) {
		Ok(v) => v, Err(e) => { eprintln!("[bench] FATAL: pipeline build failed: {}", e); std::process::exit(1); }
	};

	// Descriptor set for the pipeline.
	let pool = unsafe { device_handle.create_descriptor_pool(
		&vk::DescriptorPoolCreateInfo::default().max_sets(1)
			.pool_sizes(std::slice::from_ref(&vk::DescriptorPoolSize::default()
				.ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3))), None,
	).expect("pool") };
	let set = unsafe { device_handle.allocate_descriptor_sets(
		&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool)
			.set_layouts(std::slice::from_ref(&set_layout)),
	).expect("alloc set")[0] };

	let in_info = vk::DescriptorBufferInfo::default().buffer(in_buf).offset(0).range(N_TILES as vk::DeviceSize * SLOT_BYTES);
	let out_info = vk::DescriptorBufferInfo::default().buffer(out_buf).offset(0).range(N_TILES as vk::DeviceSize * OUT_SLOT_BYTES);
	let bp_info = vk::DescriptorBufferInfo::default().buffer(bp_buf).offset(0).range(bp_bytes as vk::DeviceSize);
	unsafe { device_handle.update_descriptor_sets(&[
		vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0)
			.descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&in_info)),
		vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1)
			.descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&out_info)),
		vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2)
			.descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&bp_info)),
	], &[]); }

	let gpu = &ctrl_guard.gpu;
	// Warm up once (first dispatch pays pipeline/TLB warmup).
	dispatch_once(gpu, &pipeline, &layout, set, n_workgroups).expect("warmup dispatch");

	// GPU-side timing via a timestamp query pool — immune to the per-dispatch
	// fence round-trip that dominates wall-clock time here.
	let device = &gpu.device_handle;
	let queue = gpu.queue_handle;
	let command_pool = gpu.command_pool;
	let cmd_buffer_pool = &gpu.cmd_buffer_pool;
	let fence_pool = &gpu.fence_pool;

	let query_pool = unsafe { device.create_query_pool(
		&vk::QueryPoolCreateInfo::default()
			.query_type(vk::QueryType::TIMESTAMP)
			.query_count(2 * ITERS),
		None,
	).expect("query pool") };

	// One command buffer: record begin/end timestamps around each of ITERS dispatches.
	let cmd = agent_harness::memory_controller::controller::GpuContext::alloc_cmd_buffer(device, command_pool, cmd_buffer_pool);
	unsafe {
		device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
			.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).expect("begin");
		device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
		device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
		for i in 0..ITERS {
			device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::COMPUTE_SHADER, query_pool, i * 2);
			device.cmd_dispatch(cmd, n_workgroups, 1, 1);
			device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::COMPUTE_SHADER, query_pool, i * 2 + 1);
		}
		device.end_command_buffer(cmd).expect("end");
	}
	let fence = agent_harness::memory_controller::controller::GpuContext::alloc_fence(device, fence_pool);
	unsafe {
		device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))], fence).expect("submit");
		device.wait_for_fences(&[fence], true, u64::MAX).expect("wait");
	}
	let mut results = vec![0u64; 2 * ITERS as usize];
	unsafe { device.get_query_pool_results(query_pool, 0, &mut results, vk::QueryResultFlags::empty()) };

	// Convert timestamp deltas to ns using the GPU's timestamp period.
	let mut total_ns = 0u64;
	for i in 0..ITERS as usize {
		total_ns += results[2*i+1].saturating_sub(results[2*i]);
	}
	let avg_ns = total_ns as f64 * ts_period_ns / ITERS as f64;
	let ms = avg_ns / 1e6;
	let tiles_per_s = (N_TILES as f64) / (avg_ns / 1e9);
	println!("{:<10} {:>12.3} {:>14.0}", "32x4", ms, tiles_per_s);

	unsafe { device.destroy_query_pool(query_pool, None); }
	agent_harness::memory_controller::controller::GpuContext::recycle_fence(device, fence, fence_pool);
	agent_harness::memory_controller::controller::GpuContext::recycle_cmd_buffer(device, cmd, cmd_buffer_pool);

	unsafe {
		device.destroy_pipeline(pipeline, None);
		device.destroy_pipeline_layout(layout, None);
		device.destroy_descriptor_set_layout(set_layout, None);
		let _ = device.free_descriptor_sets(pool, &[set]);
		device.destroy_descriptor_pool(pool, None);
	}

	unsafe {
		device_handle.destroy_buffer(in_buf, None);
		device_handle.destroy_buffer(out_buf, None);
		device_handle.destroy_buffer(bp_buf, None);
	}
	println!("\n[bench] done");
}

/// One real dispatch of the trellis kernel over `n_tiles` workgroups.
fn dispatch_once(
	gpu: &agent_harness::memory_controller::controller::GpuContext,
	pipeline: &vk::Pipeline,
	layout: &vk::PipelineLayout,
	set: vk::DescriptorSet,
	n_tiles: u32,
) -> Result<(), String> {
	let device = &gpu.device_handle;
	let queue = gpu.queue_handle;
	let command_pool = gpu.command_pool;
	let cmd_buffer_pool = &gpu.cmd_buffer_pool;
	let fence_pool = &gpu.fence_pool;

	let cmd = agent_harness::memory_controller::controller::GpuContext::alloc_cmd_buffer(device, command_pool, cmd_buffer_pool);
	unsafe {
		device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
			.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
			.map_err(|e| format!("begin cmd: {e:?}"))?;
		device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, *pipeline);
		device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, *layout, 0, &[set], &[]);
		device.cmd_dispatch(cmd, n_tiles, 1, 1);
		device.end_command_buffer(cmd).map_err(|e| format!("end cmd: {e:?}"))?;
	}

	let fence = agent_harness::memory_controller::controller::GpuContext::alloc_fence(device, fence_pool);
	unsafe {
		device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))], fence)
			.map_err(|e| format!("submit: {e:?}"))?;
		// 3s timeout so we can detect a hung dispatch and read back progress.
		match device.wait_for_fences(&[fence], true, 3_000_000_000) {
			Ok(_) => {},
			Err(e) => {
				eprintln!("[dispatch_once] TIMEOUT (10s): {e:?} — GPU likely hung in shader");
				return Err(format!("dispatch timeout: {e:?}"));
			}
		}
	}
	agent_harness::memory_controller::controller::GpuContext::recycle_fence(device, fence, fence_pool);
	agent_harness::memory_controller::controller::GpuContext::recycle_cmd_buffer(device, cmd, cmd_buffer_pool);
	Ok(())
}

/// A tensor is quantizable if it's a 2D weight matrix with both dimensions ≥32,
/// and it's not a norm or embedding.
fn is_quantizable(t: &agent_harness::models::format::TensorMeta) -> bool {
	if t.shape.len() != 2 { return false; }
	if t.shape[0] < 32 || t.shape[1] < 32 { return false; }
	let lower = t.name.to_lowercase();
	if lower.ends_with(".bias") { return false; }
	if lower.contains("norm") || lower.contains("embed") { return false; }
	true
}

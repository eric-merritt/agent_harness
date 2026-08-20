// GPU-accelerated compression using wgpu compute shaders.
//
// wgpu uses Vulkan under the hood on NVIDIA/AMD GPUs.
// The compute shader does per-element prefix/tail/sign computation on GPU.
// CPU then does grouping, truncation, and dedup on the GPU output.
//
// For a 50M-element chunk:
//   CPU: ~250ms for per-element work
//   GPU: ~2ms compute + ~50ms transfer = ~52ms total

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct Push {
    element_count: u32,
    prefix_digits: u32,
    stride: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> prefix_bits: array<u32>;
@group(0) @binding(2) var<storage, read_write> tails: array<u32>;
@group(0) @binding(3) var<storage, read_write> signs: array<u32>;
@group(0) @binding(4) var<uniform> push: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * push.stride;
    if (idx >= push.element_count) { return; }
    
    let w = input[idx];
    let abs_w = abs(w);
    let prefix_scale = pow(10.0, f32(push.prefix_digits));
    let prefix = floor(abs_w * prefix_scale) / prefix_scale;
    let tail_val = abs_w - prefix;
    let tail_scale = pow(10.0, 7.0);
    let tail_int = u32(round(tail_val * tail_scale));
    
    prefix_bits[idx] = bitcast<u32>(prefix);
    tails[idx] = tail_int;
    signs[idx] = select(0u, 1u, w < 0.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    element_count: u32,
    prefix_digits: u32,
    stride: u32,
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Check if GPU is available. Initializes the GPU on first call.
/// This adds ~700ms of Vulkan init on first call, but only once.
pub fn gpu_available() -> bool {
    GPU.get_or_init(init_gpu).is_some()
}

/// Get the GPU device/queue. Initializes on first call if needed.
pub fn gpu_device_queue() -> Option<(&'static wgpu::Device, &'static wgpu::Queue)> {
    let ctx = GPU.get_or_init(init_gpu).as_ref()?;
    Some((&ctx.device, &ctx.queue))
}

static GPU: OnceLock<Option<Arc<GpuContext>>> = OnceLock::new();
static GPU_READY: AtomicBool = AtomicBool::new(true);
static GPU_MEM_USED: AtomicU64 = AtomicU64::new(0);
static GPU_MEM_CAPACITY: AtomicU64 = AtomicU64::new(0);
const GPU_MEM_HEADROOM: u64 = 800 * 1024 * 1024;

/// Output of GPU compression pass.
pub struct GpuOutput {
    pub prefix_bits: Vec<u32>,
    pub tails: Vec<u32>,
    pub signs: Vec<u32>,
}

fn init_gpu() -> Option<Arc<GpuContext>> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Compression GPU"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: 1_073_741_824, // 1GB — sufficient for most tensors
                max_storage_buffers_per_shader_stage: 8,       // decompress shader needs 5
                ..wgpu::Limits::default()
            },
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    )).ok()?;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compression Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compression Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compression Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compression Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    let name = adapter.get_info().name;
    let backend = adapter.get_info().backend;
    log::info!("[gpu] Using: {} (backend: {:?})", name, backend);

    // Query total VRAM for memory backpressure tracking
    let vram = query_vram_bytes();
    GPU_MEM_CAPACITY.store(vram, Ordering::Relaxed);
    log::info!("[gpu] VRAM: {:.1} GB (headroom: {} MB)", vram as f64 / 1e9, GPU_MEM_HEADROOM / (1024 * 1024));

    Some(Arc::new(GpuContext { device, queue, pipeline, bind_group_layout }))
}

/// Query total VRAM in bytes from nvidia-smi. Falls back to 15 GB.
fn query_vram_bytes() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|mib| mib * 1024 * 1024)
        .unwrap_or(15 * 1024 * 1024 * 1024)
}

/// RAII guard: subtracts from GPU_MEM_USED on drop and re-enables GPU_READY if VRAM recovers.
struct MemGuard(u64);
impl Drop for MemGuard {
    fn drop(&mut self) {
        GPU_MEM_USED.fetch_sub(self.0, Ordering::Relaxed);
        let cap = GPU_MEM_CAPACITY.load(Ordering::Relaxed);
        if cap > 0 && GPU_MEM_USED.load(Ordering::Relaxed) < cap - GPU_MEM_HEADROOM {
            GPU_READY.store(true, Ordering::Relaxed);
        }
    }
}

/// Run per-element prefix/tail/sign computation on GPU.
/// Multiple workers can dispatch concurrently — wgpu's Device/Queue are Send+Sync.
/// VRAM backpressure via GPU_READY flag: when tracked allocations approach
/// capacity, the flag flips to false and workers fall back to CPU. When jobs
/// complete and free their buffers (MemGuard drop), the flag re-enables.
pub fn gpu_compute(weights: &[f32], prefix_digits: usize) -> Option<GpuOutput> {
    let gpu = GPU.get_or_init(init_gpu).as_ref()?;

    // Check VRAM backpressure flag
    if !GPU_READY.load(Ordering::Relaxed) {
        return None;
    }

    let n = weights.len();
    if n == 0 { return None; }

    let byte_size = (n * 4) as u64;
    // 7 buffers: input, prefix, tail, sign, prefix_read, tail_read, sign_read
    let job_mem = byte_size * 7;
    let prev = GPU_MEM_USED.fetch_add(job_mem, Ordering::Relaxed);
    let new_total = prev + job_mem;
    let cap = GPU_MEM_CAPACITY.load(Ordering::Relaxed);

    // If within headroom of capacity, mark not ready for other workers
    if cap > 0 && new_total > cap - GPU_MEM_HEADROOM {
        GPU_READY.store(false, Ordering::Relaxed);
    }

    // Guard ensures GPU_MEM_USED is decremented and flag is checked on drop,
    // even if a panic occurs mid-dispatch.
    let _guard = MemGuard(job_mem);
    log::debug!("[gpu] dispatching {} elements", n);

    let input_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Input"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let prefix_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Prefix"),
        size: byte_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let tail_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Tails"),
        size: byte_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let sign_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Signs"),
        size: byte_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });

    let total_groups = ((n + 255) / 256) as u32;
    let gx = total_groups.min(65535);
    let gy = (total_groups + 65534) / 65535;
    let stride = gx * 256;
    let push = PushConstants { element_count: n as u32, prefix_digits: prefix_digits as u32, stride };
    let uniform_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniforms"),
        contents: bytemuck::bytes_of(&push),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let prefix_read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Prefix Read"),
        size: byte_size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let tail_read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Tail Read"),
        size: byte_size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let sign_read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sign Read"),
        size: byte_size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });

    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compression Bind Group"),
        layout: &gpu.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: prefix_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tail_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: sign_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: uniform_buf.as_entire_binding() },
        ],
    });

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compression Encoder"),
    });

    gpu.queue.write_buffer(&input_buf, 0, bytemuck::cast_slice(weights));

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compression Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    encoder.copy_buffer_to_buffer(&prefix_buf, 0, &prefix_read, 0, byte_size);
    encoder.copy_buffer_to_buffer(&tail_buf, 0, &tail_read, 0, byte_size);
    encoder.copy_buffer_to_buffer(&sign_buf, 0, &sign_read, 0, byte_size);

    let submission = gpu.queue.submit([encoder.finish()]);
    gpu.device.poll(wgpu::Maintain::wait_for(submission));

    let prefix_bits = read_buffer(&gpu.device, &prefix_read, n);
    let tails = read_buffer(&gpu.device, &tail_read, n);
    let signs = read_buffer(&gpu.device, &sign_read, n);

    input_buf.destroy();
    prefix_buf.destroy();
    tail_buf.destroy();
    sign_buf.destroy();
    uniform_buf.destroy();
    prefix_read.destroy();
    tail_read.destroy();
    sign_read.destroy();

    // MemGuard drops here: decrements GPU_MEM_USED, re-enables GPU_READY if VRAM recovered
    Some(GpuOutput { prefix_bits, tails, signs })
}

fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::wait());
    let _ = rx.recv();
    let mut result = vec![0u32; count];
    {
        let view = slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&view);
        result.copy_from_slice(&u32_data[..count]);
    }
    buffer.unmap();
    result
}

// ── GPU Decompression Kernel ─────────────────────────────────────────────────
//
// Reconstructs f32 weight values from compressed sandbag indices + GlobalTable.
// The dictionary (prefixes + flat_tails + tail_offsets) is uploaded once and
// stays resident in VRAM. Per-tensor sandbag data is uploaded on demand.
//
// Formula: value = prefix[prefix_idx] + flat_tails[tail_offsets[prefix_idx] + tail_idx] / divisor + avg_pl
// Apply sign bit: if sign { value = -value }
//
// Sandbag packed format: u32 per element = {prefix_idx:8, tail_idx:8, sign:1, pad:7}

const DECOMPRESS_SHADER: &str = r#"
struct DecompressUniform {
    element_count: u32,
    divisor: f32,
    avg_precision_lost: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> prefixes: array<f32>;
@group(0) @binding(1) var<storage, read> flat_tails: array<u32>;
@group(0) @binding(2) var<storage, read> tail_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> sandbag: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> u: DecompressUniform;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.element_count) { return; }

    let packed = sandbag[i];
    let p_idx = packed & 0xFFu;
    let t_idx = (packed >> 8u) & 0xFFFFu;
    let sign = (packed >> 24u) & 1u;

    let prefix = prefixes[p_idx];
    let base = tail_offsets[p_idx];
    let tail = flat_tails[base + t_idx];

    var value = prefix + f32(tail) / u.divisor + u.avg_precision_lost;
    if (sign != 0u) { value = -value; }
    output[i] = value;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DecompressUniform {
    element_count: u32,
    divisor: f32,
    avg_precision_lost: f32,
    _pad: u32,
}

/// Resident GPU dictionary buffers — uploaded once, shared across all decompression dispatches.
pub struct GpuDictionary {
    prefix_buffer: wgpu::Buffer,
    tail_buffer: wgpu::Buffer,
    offset_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuDictionary {
    /// Create from a GlobalTable's CSR layout. Initializes GPU if needed.
    /// Returns None if no GPU is available.
    pub fn create_from_global_table(gt: &crate::models::dedup_count::GlobalTable) -> Option<Self> {
        let gpu = GPU.get_or_init(init_gpu).as_ref()?;
        Some(Self::create(&gpu.device, &gt.prefixes, &gt.flat_tails, &gt.tail_offsets))
    }

    /// Upload the GlobalTable CSR layout to GPU VRAM. Called once at startup.
    pub fn create(
        device: &wgpu::Device,
        prefixes: &[f32],
        flat_tails: &[u32],
        tail_offsets: &[u32],
    ) -> Self {
        let prefix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuDict prefixes"),
            contents: bytemuck::cast_slice(prefixes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let tail_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuDict flat_tails"),
            contents: bytemuck::cast_slice(flat_tails),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let offset_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuDict tail_offsets"),
            contents: bytemuck::cast_slice(tail_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Decompress Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Decompress Shader"),
            source: wgpu::ShaderSource::Wgsl(DECOMPRESS_SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Decompress Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        log::info!("GpuDictionary: resident ({} prefixes, {} tails, {} offsets)",
            prefixes.len(), flat_tails.len(), tail_offsets.len());

        Self { prefix_buffer, tail_buffer, offset_buffer, pipeline, bind_group_layout }
    }

    /// Dispatch the decompression kernel for one chunk.
    /// `sandbag_packed` is u32 per element: {prefix_idx:8, tail_idx:8, sign:1, pad:7}.
    /// Returns reconstructed f32 values.
    pub fn decompress(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sandbag_packed: &[u32],
        element_count: usize,
        divisor: f32,
        avg_precision_lost: f32,
    ) -> Vec<f32> {
        let sandbag_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Decompress sandbag"),
            contents: bytemuck::cast_slice(sandbag_packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Decompress output"),
            size: (element_count * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniform = DecompressUniform {
            element_count: element_count as u32,
            divisor,
            avg_precision_lost,
            _pad: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Decompress uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Decompress bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.prefix_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.tail_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.offset_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sandbag_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: uniform_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Decompress encoder"),
        });

        let (dispatch_x, dispatch_y) = dispatch_dims(element_count);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Decompress pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Copy output to readback buffer
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Decompress readback"),
            size: (element_count * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, (element_count * 4) as wgpu::BufferAddress);

        queue.submit(std::iter::once(encoder.finish()));

        // Read back
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::wait());

        let mut result = vec![0.0f32; element_count];
        if rx.recv() == Ok(Ok(())) {
            let view = slice.get_mapped_range();
            let f32_data: &[f32] = bytemuck::cast_slice(&view);
            result.copy_from_slice(&f32_data[..element_count]);
        }
        readback.unmap();

        result
    }
}

fn dispatch_dims(count: usize) -> (u32, u32) {
    let groups = (count + 255) / 256;
    let x = (groups as u32).min(65535);
    let y = ((groups + 65534) / 65535) as u32 + 1;
    (x, y)
}

//! CPU vs GPU sandbag encoding, same tensors, same output.
//!
//! Both paths write to /dev/null so the comparison is encode throughput rather
//! than disk. The GPU number includes submit, fence wait and readback — the full
//! round trip, not just shader time — because that is what a conversion pays.
//!
//! Skips the GPU group when no Vulkan device is usable.

use agent_harness::memory_controller::controller::init_global_controller;
use agent_harness::memory_controller::gpu_mem_op::init_gpu;
use agent_harness::models::format::GgmlType;
use agent_harness::models::quantize::{quantize_cpu, quantize_gpu};
use agent_harness::models::tensor::Tensor;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::path::Path;
use std::sync::OnceLock;

/// bf16 tensors totalling roughly `target` weights.
fn build_model(target: usize) -> (Vec<Tensor>, Vec<u8>) {
	let (rows, cols) = (1024usize, 1024usize);
	let per = rows * cols;
	let count = target.div_ceil(per);

	let mut raw = Vec::with_capacity(count * per * 2);
	let mut tensors = Vec::with_capacity(count);
	let mut s = 0x2545_F491_4F6C_DD1Du64;

	for ti in 0..count {
		let offset = raw.len() as u64;
		for _ in 0..per {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			let v = ((s >> 40) as f32 / 16_777_216.0 - 0.5) * 0.08;
			raw.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
		}
		tensors.push(Tensor::new(
			format!("blk.{ti}.weight"),
			vec![rows, cols],
			GgmlType::BF16,
			Tensor::classify("weight"),
			offset,
			(per * 2) as u64,
		));
	}
	(tensors, raw)
}

static GPU_OK: OnceLock<bool> = OnceLock::new();

fn gpu_ready() -> bool {
	*GPU_OK.get_or_init(|| {
		match init_gpu() {
			Ok((entry, instance, pd, device, queue, alloc)) => {
				// `entry` and `instance` are stored in the global Vulkan guard so the
				// libvulkan library stays loaded and the device's VFN table stays valid.
				match init_global_controller(entry, &instance, pd, device.clone(), queue, alloc) {
					Ok(()) => true,
					Err(e) => e.contains("already initialized"),
				}
			}
			Err(e) => {
				eprintln!("no Vulkan device ({e}); skipping GPU group");
				false
			}
		}
	})
}

fn bench_cpu_vs_gpu(c: &mut Criterion) {
	let sink = Path::new("/dev/null");
	let have_gpu = gpu_ready();

	for &weights in &[1 << 20, 1 << 22] {
		let (tensors, raw) = build_model(weights);
		let n: usize = tensors.iter().map(|t| t.elem_count).sum();

		let mut g = c.benchmark_group(format!("quantize/{}M_weights", n / (1 << 20)));
		g.throughput(Throughput::Elements(n as u64));
		g.sample_size(20);

		g.bench_function("cpu_rayon", |b| {
			b.iter(|| quantize_cpu(black_box(&tensors), black_box(&raw), sink).unwrap());
		});

		if have_gpu {
			// Warm-up: first call pages the model into the arena and commits pages.
			if let Err(e) = quantize_gpu(&tensors, &raw, sink) {
				eprintln!("GPU quantize unavailable ({e}); skipping");
				g.finish();
				continue;
			}
			g.bench_function("gpu_vulkan", |b| {
				b.iter(|| quantize_gpu(black_box(&tensors), black_box(&raw), sink).unwrap());
			});
		}

		g.finish();
	}
}

criterion_group!(benches, bench_cpu_vs_gpu);
criterion_main!(benches);

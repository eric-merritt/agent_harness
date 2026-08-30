// Unified virtual device — abstracts GPU VRAM + CPU RAM into one address space.
//
// Core principles:
//   1. Areas are planned before data is loaded (plan_region → load_region)
//   2. Memory stays mapped for the device's lifetime — no remapping, no reallocation
//   3. Zero-copy casting via bytemuck: cast_region::<f32>() reinterprets bytes in-place
//   4. Best-fit distribution: the device decides placement, but consumers see one API
//
// Usage:
//   let mut dev = VirtualDevice::new().with_cpu(8_000_000_000).with_gpu()?.build()?;
//   let id = dev.plan_region("prefixes", 1024, 4, PlacementHint::Gpu)?;
//   dev.load_region(id, &prefix_bytes)?;
//   let view: &[f32] = dev.cast_region(id)?;
//   let (buf, off, len) = dev.gpu_binding(id)?;

pub mod controller;
pub mod cpu_mem_op;
// pub mod gpu_mem_op; // removed — operations live in controller.rs
pub mod virtual_tensor_arena;

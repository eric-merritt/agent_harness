//! GPU dispatch and bucket readback helpers.
//!
//! Owns Vulkan shader dispatch (command buffers, descriptor sets, fences)
//! and downloading/parsing bucket output from the GPU arena.

use crate::memory_controller::controller::{MemoryController, GpuContext};
use crate::models::compression::gpu_helpers::{BlockQuantizeOutput, GpuQuantizeOutput, extract_tails_from_u32};
use crate::models::formats::sandbag::BucketEntry;
use ash::vk;

/// Records the pipeline state, updates descriptors, binds push constants,
/// and dispatches the workgroups to the compute queue.
pub unsafe fn dispatch_quantize_shader(
	ctrl: &MemoryController,
	buf: vk::Buffer,
	offset: vk::DeviceSize,
	total_elements: u32,
	prefix_digits: u32,
) -> Result<(GpuQuantizeOutput, vk::Fence, ash::Device), String> {
	let device = &ctrl.gpu.device_handle;

	// 1. Get command buffer from pool (recycled or fresh)
	let cmd_buffer = GpuContext::alloc_cmd_buffer(
		device,
		ctrl.gpu.command_pool,
		&ctrl.gpu.cmd_buffer_pool,
	);

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

	// 5. Calculate local work groups thread counts (clamped ceiling)
	let work_groups_x = (total_elements + 255) / 256;
	device.cmd_dispatch(cmd_buffer, work_groups_x, 1, 1);

	// 6. End command recording pass
	device
		.end_command_buffer(cmd_buffer)
		.map_err(|e| format!("Failed to end command buffer: {:?}", e))?;

	// 7. Get fence from pool (recycled or fresh)
	let fence = GpuContext::alloc_fence(device, &ctrl.gpu.fence_pool);

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
pub fn read_bucket_output(controller: &MemoryController) -> Result<GpuQuantizeOutput, ()> {
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
			prefix: bucket_idx as u8,
			tails: extract_tails_from_u32(packed_val),
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

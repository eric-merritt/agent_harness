//! Format converters: safetensors, gguf, pt → sandbag.
//!
//! Each function imports its own format reader, calls quantize, and writes
//! the sandbag output. That's it.

use crate::models::format::*;
use crate::models::quantize::*;
use crate::models::tensor::*;
use std::collections::HashMap;

/// Convert safetensors → sandbag.
pub fn convert_safetensors_to_sandbag(
	src_path: &std::path::Path,
	dst_path: &std::path::Path,
) -> Result<(), String> {
	log::info!(
		"Converting safetensors {} → sandbag {}",
		src_path.display(),
		dst_path.display()
	);

	let raw = std::fs::read(src_path).map_err(|e| format!("Read failed: {}", e))?;
	let header = parse_safetensors_header(&raw)?;

	// Build tensor descriptors from the header
	let tensors: Vec<Tensor> = header
		.tensors
		.iter()
		.map(|(name, info)| {
			Tensor::new(
				name.clone(),
				info.shape.iter().map(|&d| d as usize).collect(),
				parse_dtype(&info.dtype),
				Tensor::classify(name),
				info.data_offsets.0,
				(info.data_offsets.1 - info.data_offsets.0) as u64,
			)
		})
		.collect();

	// Quantize and write
	quantize_cpu(&tensors, &raw, dst_path)
}

/// Convert GGUF → sandbag.
pub fn convert_gguf_to_sandbag(
	src_path: &std::path::Path,
	dst_path: &std::path::Path,
) -> Result<(), String> {
	log::info!(
		"Converting GGUF {} → sandbag {}",
		src_path.display(),
		dst_path.display()
	);

	let raw = std::fs::read(src_path).map_err(|e| format!("Read failed: {}", e))?;
	let header = parse_gguf_header(&raw)?;

	let tensors: Vec<Tensor> = header
		.tensor_info
		.iter()
		.map(|t| {
			let elem_count: u64 = t.shape.iter().product();
			Tensor::new(
				t.name.clone(),
				t.shape.iter().map(|&d| d as usize).collect(),
				t.dtype,
				Tensor::classify(&t.name),
				t.data_offset + t.alignment_padding,
				t.dtype.tensor_bytes(elem_count),
			)
		})
		.collect();

	quantize_cpu(&tensors, &raw, dst_path)
}

/// Convert PyTorch (.pt / .pth) → sandbag.
pub fn convert_pt_to_sandbag(
	src_path: &std::path::Path,
	dst_path: &std::path::Path,
) -> Result<(), String> {
	log::info!(
		"Converting PT {} → sandbag {}",
		src_path.display(),
		dst_path.display()
	);

	let raw = std::fs::read(src_path).map_err(|e| format!("Read failed: {}", e))?;
	let infos = parse_pickle_tensors(&raw)?;

	let tensors: Vec<Tensor> = infos
		.iter()
		.map(|info| {
			Tensor::new(
				info.name.clone(),
				info.shape.iter().map(|&d| d as usize).collect(),
				parse_dtype(&info.dtype),
				Tensor::classify(&info.name),
				info.data_offset,
				info.data_len,
			)
		})
		.collect();

	quantize_cpu(&tensors, &raw, dst_path)
}

// ---------------------------------------------------------------------------
// Minimal parsers (full parsing lives in the loader layer)
// ---------------------------------------------------------------------------

fn parse_dtype(s: &str) -> GgmlType {
	let lower = s.to_lowercase();
	if lower.contains("float32") || lower.contains("fp32") || lower.contains("f32") {
		return GgmlType::F32;
	}
	if lower.contains("float16") || lower.contains("fp16") || lower.contains("f16") {
		return GgmlType::F16;
	}
	if lower.contains("bfloat") || lower.contains("bf16") {
		return GgmlType::BF16;
	}
	GgmlType::F32
}

fn parse_safetensors_header(data: &[u8]) -> Result<SafeTensorsHeader, String> {
	// First 8 bytes = u64 LE length of JSON header
	if data.len() < 8 {
		return Err("File too small for safetensors header".into());
	}
	let json_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
	let json_start = 8;
	let json_end = json_start + json_len;
	if json_end > data.len() {
		return Err("Truncated safetensors header".into());
	}
	let json_str = std::str::from_utf8(&data[json_start..json_end])
		.map_err(|e| format!("UTF-8 decode: {}", e))?;
	let header: SafeTensorsHeader =
		serde_json::from_str(json_str).map_err(|e| format!("JSON parse: {}", e))?;
	Ok(header)
}

fn parse_gguf_header(data: &[u8]) -> Result<GgufHeader, String> {
	parse_gguf(data)
}

fn parse_pickle_tensors(_data: &[u8]) -> Result<Vec<PickleTensorInfo>, String> {
	// Minimal pickle stream reader — production loader fills in offsets.
	// Placeholder that returns empty list; replace with real parser.
	Ok(Vec::new())
}

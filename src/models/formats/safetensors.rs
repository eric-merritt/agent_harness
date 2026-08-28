use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::super::adapters::ModelAdapter;
use super::super::formats::gguf;
use crate::augment::augment::{TensorDescriptor, TensorDtype};

pub struct SafetensorsFile {
	pub header: SafetensorsHeader,
	pub tensor_raw: Vec<u8>,
}

impl SafetensorsFile {
	pub fn default_alignment() -> usize {
		8
	}

	pub fn padding_needed(current_offset: usize, alignment: usize) -> usize {
		let remainder = current_offset % alignment;
		if remainder == 0 {
			0
		} else {
			alignment - remainder
		}
	}
}

impl ModelAdapter for SafetensorsFile {
	fn model_name(&self) -> &str {
		self.header
			.kv_meta
			.as_ref()
			.and_then(|m| m.get("format"))
			.map(|s| s.as_str())
			.unwrap_or("safetensors")
	}

	fn tensor_count(&self) -> usize {
		self.header.tensor_info.len()
	}

	fn total_bytes(&self) -> u64 {
		self.header
			.tensor_info
			.values()
			.map(|t| (t.data_offsets[1] - t.data_offsets[0]) as u64)
			.sum()
	}

	fn tensor_at(&self, index: usize) -> Option<TensorDescriptor> {
		self.header
			.tensor_info
			.iter()
			.nth(index)
			.map(|(name, info)| {
				let byte_size = (info.data_offsets[1] - info.data_offsets[0]) as u64;
				TensorDescriptor {
					name: name.clone(),
					shape: info.shape.clone(),
					dtype: info.dtype.into(),
					byte_offset: info.data_offsets[0] as u64,
					byte_size,
					layer_index: TensorDescriptor::parse_layer_index(name),
				}
			})
	}

	fn find_tensor(&self, name: &str) -> Option<TensorDescriptor> {
		self.header.tensor_info.get(name).map(|info| {
			let byte_size = (info.data_offsets[1] - info.data_offsets[0]) as u64;
			TensorDescriptor {
				name: name.to_string(),
				shape: info.shape.clone(),
				dtype: info.dtype.into(),
				byte_offset: info.data_offsets[0] as u64,
				byte_size,
				layer_index: TensorDescriptor::parse_layer_index(name),
			}
		})
	}

	fn alignment(&self) -> usize {
		Self::default_alignment()
	}

	fn raw_data(&self, tensor_name: &str) -> Option<&[u8]> {
		self.header.tensor_info.get(tensor_name).map(|info| {
			let start = info.data_offsets[0];
			let end = info.data_offsets[1];
			&self.tensor_raw[start..end]
		})
	}
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SafetensorInfo {
	pub dtype: SafetensorsDtype,
	pub shape: Vec<usize>,
	pub data_offsets: [usize; 2],
}

impl SafetensorInfo {
	/// Number of elements in this tensor.
	pub fn element_count(&self) -> usize {
		if self.shape.is_empty() {
			1
		} else {
			self.shape.iter().product()
		}
	}

	/// Byte size of this tensor's raw data.
	pub fn byte_size(&self) -> usize {
		self.data_offsets[1] - self.data_offsets[0]
	}

	/// Start offset within the data section.
	pub fn data_offset(&self) -> usize {
		self.data_offsets[0]
	}
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SafetensorsHeader {
	#[serde(rename = "__metadata__", default)]
	pub kv_meta: Option<HashMap<String, String>>,
	#[serde(flatten)]
	pub tensor_info: HashMap<String, SafetensorInfo>,
}

impl SafetensorsHeader {
	/// Parse just the header from a safetensors file on disk.
	/// Returns (header, data_section_start_offset).
	pub fn parse_from_file(
		path: &std::path::Path,
	) -> Result<(Self, usize), Box<dyn std::error::Error>> {
		use std::io::Read;
		let mut f = std::fs::File::open(path)?;
		let mut len_buf = [0u8; 8];
		f.read_exact(&mut len_buf)?;
		let header_len = u64::from_le_bytes(len_buf) as usize;
		let mut hdr = vec![0u8; header_len];
		f.read_exact(&mut hdr)?;
		let header: Self = serde_json::from_slice(&hdr)?;
		Ok((header, 8 + header_len))
	}

	/// Parse the header from raw file bytes.
	/// Returns (header, data_section_start_offset).
	pub fn parse_from_bytes(data: &[u8]) -> Result<(Self, usize), Box<dyn std::error::Error>> {
		if data.len() < 8 {
			return Err("file too small".into());
		}
		let header_len = u64::from_le_bytes(data[0..8].try_into()?) as usize;
		if data.len() < 8 + header_len {
			return Err("header extends past file".into());
		}
		let header: Self = serde_json::from_slice(&data[8..8 + header_len])?;
		Ok((header, 8 + header_len))
	}

	/// Sorted tensor names for deterministic ordering.
	pub fn sorted_tensors(&self) -> Vec<(&String, &SafetensorInfo)> {
		let mut entries: Vec<_> = self.tensor_info.iter().collect();
		entries.sort_by(|a, b| a.0.cmp(b.0));
		entries
	}
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum SafetensorsDtype {
	BOOL,
	U8,
	I8,
	I16,
	U16,
	I32,
	U32,
	F16,
	BF16,
	F32,
	F64,
	I64,
	U64,
}

impl SafetensorsDtype {
	/// Bytes per element for this dtype.
	pub fn bytes_per_element(&self) -> usize {
		match self {
			SafetensorsDtype::BOOL | SafetensorsDtype::U8 | SafetensorsDtype::I8 => 1,
			SafetensorsDtype::I16
			| SafetensorsDtype::U16
			| SafetensorsDtype::F16
			| SafetensorsDtype::BF16 => 2,
			SafetensorsDtype::I32 | SafetensorsDtype::U32 | SafetensorsDtype::F32 => 4,
			SafetensorsDtype::I64 | SafetensorsDtype::U64 | SafetensorsDtype::F64 => 8,
		}
	}

	/// Dequantize raw bytes to f32 for this dtype.
	pub fn dequantize_to_f32(&self, raw: &[u8], n_elems: usize) -> Vec<f32> {
		match self {
			SafetensorsDtype::F32 => raw
				.chunks_exact(4)
				.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
				.collect(),
			SafetensorsDtype::F16 => raw
				.chunks_exact(2)
				.map(|c| gguf::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
				.collect(),
			SafetensorsDtype::BF16 => raw
				.chunks_exact(2)
				.map(|c| gguf::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
				.collect(),
			_ => vec![0.0; n_elems],
		}
	}
}

impl From<SafetensorsDtype> for TensorDtype {
	fn from(d: SafetensorsDtype) -> Self {
		match d {
			SafetensorsDtype::F32 => TensorDtype::F32,
			SafetensorsDtype::F16 => TensorDtype::F16,
			SafetensorsDtype::BF16 => TensorDtype::BF16,
			SafetensorsDtype::I8 => TensorDtype::I8,
			SafetensorsDtype::I16 => TensorDtype::I16,
			SafetensorsDtype::I32 => TensorDtype::I32,
			SafetensorsDtype::I64 => TensorDtype::I64,
			SafetensorsDtype::U8 => TensorDtype::U8,
			SafetensorsDtype::BOOL => TensorDtype::Bool,
			SafetensorsDtype::F64
			| SafetensorsDtype::U16
			| SafetensorsDtype::U32
			| SafetensorsDtype::U64 => TensorDtype::F32,
		}
	}
}

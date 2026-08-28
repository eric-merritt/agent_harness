use std::collections::HashMap;

use super::super::adapters::ModelAdapter;
use crate::augment::augment::{TensorDescriptor, TensorDtype};

pub struct PtFile {
	pub kv_meta: HashMap<String, String>,
	pub tensor_info: HashMap<String, PtTensorInfo>,
	pub tensor_raw: Vec<u8>,
}

impl PtFile {
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

impl ModelAdapter for PtFile {
	fn model_name(&self) -> &str {
		self.kv_meta
			.get("model_name")
			.map(|s| s.as_str())
			.unwrap_or("pytorch")
	}

	fn tensor_count(&self) -> usize {
		self.tensor_info.len()
	}

	fn total_bytes(&self) -> u64 {
		self.tensor_info
			.values()
			.map(|t| (t.data_offsets[1] - t.data_offsets[0]) as u64)
			.sum()
	}

	fn tensor_at(&self, index: usize) -> Option<TensorDescriptor> {
		self.tensor_info.iter().nth(index).map(|(name, info)| {
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
		self.tensor_info.get(name).map(|info| {
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
		self.tensor_info
			.get(tensor_name)
			.map(|info| &self.tensor_raw[info.data_offsets[0]..info.data_offsets[1]])
	}
}

#[derive(Debug, Clone, PartialEq)]
pub struct PtTensorInfo {
	pub name: String,
	pub shape: Vec<usize>,
	pub dtype: PtDtype,
	pub data_offsets: [usize; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtDtype {
	Bool,
	Byte,
	Char,
	Short,
	Int,
	Long,
	Half,
	BFloat16,
	Float,
	Double,
}

impl From<PtDtype> for TensorDtype {
	fn from(d: PtDtype) -> Self {
		match d {
			PtDtype::Float => TensorDtype::F32,
			PtDtype::Half => TensorDtype::F16,
			PtDtype::BFloat16 => TensorDtype::BF16,
			PtDtype::Byte => TensorDtype::U8,
			PtDtype::Char => TensorDtype::I8,
			PtDtype::Short => TensorDtype::I16,
			PtDtype::Int => TensorDtype::I32,
			PtDtype::Long => TensorDtype::I64,
			PtDtype::Bool => TensorDtype::Bool,
			PtDtype::Double => TensorDtype::F32,
		}
	}
}

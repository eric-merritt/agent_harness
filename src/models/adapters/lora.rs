use std::collections::HashMap;

use super::ModelAdapter;
use crate::augment::augment::{TensorDescriptor, TensorDtype};

/// A LoRA adapter file (safetensors format with A/B matrices).
/// Implements ModelAdapter so it can be plugged into the AugmentBus.
pub struct LoRAAdapter {
	pub base_model: String,
	pub adapter_path: String,
	pub rank: u32,
	pub target_modules: Vec<String>,
	pub alpha: f32,
	pub dropout: f32,
	pub tensors: HashMap<String, LoRATensor>,
	pub tensor_raw: Vec<u8>,
}

/// A single LoRA tensor (either lora_A or lora_B matrix).
#[derive(Debug, Clone)]
pub struct LoRATensor {
	pub name: String,
	pub shape: Vec<usize>,
	pub dtype: LoRADtype,
	pub data_offsets: [usize; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoRADtype {
	F16,
	BF16,
	F32,
	I8,
	I16,
	I32,
	I64,
	U8,
	Bool,
}

impl From<LoRADtype> for TensorDtype {
	fn from(d: LoRADtype) -> Self {
		match d {
			LoRADtype::F32 => TensorDtype::F32,
			LoRADtype::F16 => TensorDtype::F16,
			LoRADtype::BF16 => TensorDtype::BF16,
			LoRADtype::I8 => TensorDtype::I8,
			LoRADtype::I16 => TensorDtype::I16,
			LoRADtype::I32 => TensorDtype::I32,
			LoRADtype::I64 => TensorDtype::I64,
			LoRADtype::U8 => TensorDtype::U8,
			LoRADtype::Bool => TensorDtype::Bool,
		}
	}
}

impl LoRAAdapter {
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

	/// Get the A and B matrices for a given target module.
	pub fn get_ab_pair(&self, module: &str) -> Option<(&LoRATensor, &LoRATensor)> {
		let a_name = format!("lora_A.{}", module);
		let b_name = format!("lora_B.{}", module);
		let a = self.tensors.get(&a_name)?;
		let b = self.tensors.get(&b_name)?;
		Some((a, b))
	}

	/// List all target modules that have A/B pairs loaded.
	pub fn loaded_modules(&self) -> Vec<&str> {
		self.tensors
			.keys()
			.filter(|n| n.starts_with("lora_A."))
			.filter_map(|n| n.strip_prefix("lora_A."))
			.collect()
	}
}

impl ModelAdapter for LoRAAdapter {
	fn model_name(&self) -> &str {
		&self.base_model
	}

	fn tensor_count(&self) -> usize {
		self.tensors.len()
	}

	fn total_bytes(&self) -> u64 {
		self.tensors
			.values()
			.map(|t| (t.data_offsets[1] - t.data_offsets[0]) as u64)
			.sum()
	}

	fn tensor_at(&self, index: usize) -> Option<TensorDescriptor> {
		self.tensors.iter().nth(index).map(|(name, info)| {
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
		self.tensors.get(name).map(|info| {
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
		self.tensors
			.get(tensor_name)
			.map(|info| &self.tensor_raw[info.data_offsets[0]..info.data_offsets[1]])
	}
}

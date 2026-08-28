// Raw memory buffer with device abstraction.
// CPU = system RAM, GPU = Vulkan-allocated VRAM.
// Buffer owns the allocation; Tensor references pages in the VirtualTensorArena.

use crate::types::shape::Shape;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
	CPU,
	GPU,
}

pub struct Buffer {
	pub device: Device,
	pub shape: Shape,
	pub data: Vec<u8>,
}

impl Buffer {
	pub fn new(device: Device, shape: Shape, data: Vec<u8>) -> Self {
		Self {
			device,
			shape,
			data,
		}
	}

	pub fn byte_size(&self) -> usize {
		self.data.len()
	}
}

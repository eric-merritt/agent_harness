// Tensor — a view into the VirtualTensorArena.
//
// A tensor is a typed window over the arena's pages. It knows its dtype, shape,
// strides, and where it starts in the arena. Slicing creates new views without
// copying data — zero-copy.

use crate::types::dtype::DataType;
use crate::types::shape::{Dim, Shape};
use smallvec::SmallVec;

pub struct Tensor {
	/// The underlying numerical data type (F32, F16, BF16, INT4, Q8_0, etc.)
	pub dtype: DataType,

	/// Multi-dimensional shape descriptor.
	pub shape: Shape,

	/// Stride map tracking memory spacing. Crucial for non-contiguous views/slicing.
	pub strides: SmallVec<[usize; 4]>,

	/// Starting page index inside the VirtualTensorArena.
	pub start_page_index: usize,

	/// Byte offset within the starting page (handles non-page-aligned slicing).
	pub byte_offset_in_arena: usize,

	/// Whether memory is contiguous (row-major) for fast block operations.
	pub is_contiguous: bool,
}

impl Tensor {
	/// Instantiate a new contiguous tensor window over the virtual arena pages.
	pub fn new(
		dtype: DataType,
		shape: Shape,
		start_page_index: usize,
		byte_offset_in_arena: usize,
	) -> Option<Self> {
		let strides = shape.compute_strides()?;
		Some(Self {
			dtype,
			shape,
			strides,
			start_page_index,
			byte_offset_in_arena,
			is_contiguous: true,
		})
	}

	/// Create a shallow slice/view over an existing tensor WITHOUT copying bytes.
	/// Zero-copy: adjusts the byte offset and shape, reuses the same arena pages.
	pub fn slice_view(&self, dim_index: usize, start: usize, end: usize) -> Option<Self> {
		if dim_index >= self.shape.rank() {
			return None;
		}
		let dim_size = match self.shape.get_dimension(dim_index)? {
			Dim::Known(size) => size,
			Dim::Unknown => return None,
		};
		if start >= end || end > dim_size {
			return None;
		}

		let element_size = self.dtype.byte_size();
		let stride_multiplier = self.strides[dim_index];
		let byte_shift = start * stride_multiplier * element_size;

		let mut new_dims = SmallVec::new();
		for i in 0..self.shape.rank() {
			if i == dim_index {
				new_dims.push(Dim::Known(end - start));
			} else {
				new_dims.push(self.shape.get_dimension(i)?);
			}
		}

		Some(Self {
			dtype: self.dtype,
			shape: Shape::new(new_dims),
			strides: self.strides.clone(),
			start_page_index: self.start_page_index,
			byte_offset_in_arena: self.byte_offset_in_arena + byte_shift,
			is_contiguous: false,
		})
	}

	/// Total bytes this tensor occupies in the arena.
	pub fn byte_size(&self) -> usize {
		match self.shape.numel() {
			Some(n) => self.dtype.total_bytes(n),
			None => 0,
		}
	}

	/// Element count (if shape is fully known).
	pub fn numel(&self) -> Option<usize> {
		self.shape.numel()
	}
}

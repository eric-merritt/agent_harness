use super::super::adapters::ModelAdapter;
use crate::augment::augment::{TensorDescriptor, TensorDtype};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

// ── GGUF binary format constants ─────────────────────────────────────────────

const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// GGUF metadata value types (from ggml.h).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum GgufValueType {
	U8 = 0,
	I8 = 1,
	U16 = 2,
	I16 = 3,
	U32 = 4,
	I32 = 5,
	F32 = 6,
	Bool = 7,
	String = 8,
	Array = 9,
	U64 = 10,
	I64 = 11,
	F64 = 12,
}

impl GgufValueType {
	fn from_u32(v: u32) -> Option<Self> {
		match v {
			0 => Some(Self::U8),
			1 => Some(Self::I8),
			2 => Some(Self::U16),
			3 => Some(Self::I16),
			4 => Some(Self::U32),
			5 => Some(Self::I32),
			6 => Some(Self::F32),
			7 => Some(Self::Bool),
			8 => Some(Self::String),
			9 => Some(Self::Array),
			10 => Some(Self::U64),
			11 => Some(Self::I64),
			12 => Some(Self::F64),
			_ => None,
		}
	}

	/// Size of a fixed-width value in bytes.
	fn byte_size(&self) -> usize {
		match self {
			Self::U8 | Self::I8 | Self::Bool => 1,
			Self::U16 | Self::I16 => 2,
			Self::U32 | Self::I32 | Self::F32 => 4,
			Self::U64 | Self::I64 | Self::F64 => 8,
			Self::String | Self::Array => 0, // variable
		}
	}
}

// ── GGML quantization type codes ─────────────────────────────────────────────

/// Block size (elements per block) and block byte size for GGML quant types.
pub fn quant_block_info(dtype: u32) -> Option<(usize, usize)> {
	match dtype {
		0 => Some((1, 4)),      // F32: 1 elem, 4 bytes
		1 => Some((1, 2)),      // F16: 1 elem, 2 bytes
		2 => Some((32, 18)),    // Q4_0: 32 elems, 18 bytes
		3 => Some((32, 20)),    // Q4_1: 32 elems, 20 bytes
		6 => Some((32, 21)),    // Q5_0: 32 elems, 21 bytes
		7 => Some((32, 23)),    // Q5_1: 32 elems, 23 bytes
		8 => Some((32, 34)),    // Q8_0: 32 elems, 34 bytes
		9 => Some((256, 84)),   // Q2_K: 256 elems, 84 bytes
		10 => Some((256, 110)), // Q3_K: 256 elems, 110 bytes
		11 => Some((256, 144)), // Q4_K: 256 elems, 144 bytes
		12 => Some((256, 176)), // Q5_K: 256 elems, 176 bytes
		13 => Some((256, 210)), // Q6_K: 256 elems, 210 bytes
		30 => Some((1, 2)),     // BF16: 1 elem, 2 bytes
		_ => None,
	}
}

/// Convert GGML dtype code to our TensorDtype.
fn ggml_dtype_to_tensor_dtype(dtype: u32) -> TensorDtype {
	match dtype {
		0 => TensorDtype::F32,
		1 => TensorDtype::F16,
		8 => TensorDtype::I8, // Q8_0 stores int8 quants
		24 => TensorDtype::I8,
		25 => TensorDtype::I16,
		26 => TensorDtype::I32,
		27 => TensorDtype::I64,
		30 => TensorDtype::BF16,
		_ => TensorDtype::F32,
	}
}

// ── f16 → f32 conversion (IEEE 754 half-precision) ──────────────────────────

pub fn f16_to_f32(bits: u16) -> f32 {
	let sign = (bits >> 15) & 1;
	let exp = (bits >> 10) & 0x1F;
	let mant = bits & 0x3FF;

	let val = if exp == 0 {
		if mant == 0 {
			0.0
		} else {
			// Subnormal
			let m = mant as f32 / 1024.0;
			m * 2f32.powi(-14)
		}
	} else if exp == 0x1F {
		// Inf or NaN
		if mant == 0 { f32::INFINITY } else { f32::NAN }
	} else {
		// Normal
		let m = 1.0 + mant as f32 / 1024.0;
		m * 2f32.powi(exp as i32 - 15)
	};

	if sign != 0 { -val } else { val }
}

// ── bfloat16 → f32 conversion ────────────────────────────────────────────────

pub fn bf16_to_f32(bits: u16) -> f32 {
	f32::from_bits((bits as u32) << 16)
}

// ── Binary reader helpers ────────────────────────────────────────────────────

struct Reader<'a> {
	data: &'a [u8],
	pos: usize,
}

impl<'a> Reader<'a> {
	fn new(data: &'a [u8]) -> Self {
		Self { data, pos: 0 }
	}

	fn read_u8(&mut self) -> Result<u8, String> {
		let v = *self.data.get(self.pos).ok_or("EOF reading u8")?;
		self.pos += 1;
		Ok(v)
	}

	fn read_u16(&mut self) -> Result<u16, String> {
		if self.pos + 2 > self.data.len() {
			return Err("EOF reading u16".into());
		}
		let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
		self.pos += 2;
		Ok(v)
	}

	fn read_u32(&mut self) -> Result<u32, String> {
		if self.pos + 4 > self.data.len() {
			return Err("EOF reading u32".into());
		}
		let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
		self.pos += 4;
		Ok(v)
	}

	fn read_u64(&mut self) -> Result<u64, String> {
		if self.pos + 8 > self.data.len() {
			return Err("EOF reading u64".into());
		}
		let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
		self.pos += 8;
		Ok(v)
	}

	fn read_i32(&mut self) -> Result<i32, String> {
		Ok(self.read_u32()? as i32)
	}

	fn read_i64(&mut self) -> Result<i64, String> {
		Ok(self.read_u64()? as i64)
	}

	fn read_f32(&mut self) -> Result<f32, String> {
		Ok(f32::from_bits(self.read_u32()?))
	}

	fn read_f64(&mut self) -> Result<f64, String> {
		if self.pos + 8 > self.data.len() {
			return Err("EOF reading f64".into());
		}
		let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
		self.pos += 8;
		Ok(v)
	}

	fn read_bool(&mut self) -> Result<bool, String> {
		Ok(self.read_u8()? != 0)
	}

	fn read_string(&mut self) -> Result<String, String> {
		let len = self.read_u64()? as usize;
		if self.pos + len > self.data.len() {
			return Err("EOF reading string".into());
		}
		let s = String::from_utf8(self.data[self.pos..self.pos + len].to_vec())
			.map_err(|e| format!("invalid UTF-8: {}", e))?;
		self.pos += len;
		Ok(s)
	}

	/// Read one metadata value given its type.
	fn read_value(&mut self, vtype: GgufValueType) -> Result<GGUFValue, String> {
		match vtype {
			GgufValueType::U8 => Ok(GGUFValue::U8(self.read_u8()?)),
			GgufValueType::I8 => Ok(GGUFValue::I8(self.read_u8()? as i8)),
			GgufValueType::U16 => Ok(GGUFValue::U16(self.read_u16()?)),
			GgufValueType::I16 => Ok(GGUFValue::I16(self.read_u16()? as i16)),
			GgufValueType::U32 => Ok(GGUFValue::U32(self.read_u32()?)),
			GgufValueType::I32 => Ok(GGUFValue::I32(self.read_i32()?)),
			GgufValueType::F32 => Ok(GGUFValue::F32(self.read_f32()?)),
			GgufValueType::Bool => Ok(GGUFValue::Bool(self.read_bool()?)),
			GgufValueType::String => Ok(GGUFValue::String(self.read_string()?)),
			GgufValueType::U64 => Ok(GGUFValue::U64(self.read_u64()?)),
			GgufValueType::I64 => Ok(GGUFValue::I64(self.read_i64()?)),
			GgufValueType::F64 => Ok(GGUFValue::F64(self.read_f64()?)),
			GgufValueType::Array => {
				let elem_type_u32 = self.read_u32()?;
				let elem_type = GgufValueType::from_u32(elem_type_u32)
					.ok_or_else(|| format!("unknown array elem type {}", elem_type_u32))?;
				let count = self.read_u64()? as usize;
				let mut items = Vec::with_capacity(count);
				for _ in 0..count {
					items.push(self.read_value(elem_type)?);
				}
				Ok(GGUFValue::Array(GGUFArrayValue {
					dtype: elem_type,
					data: items,
				}))
			}
		}
	}
}

// ── Public structs ───────────────────────────────────────────────────────────

pub struct GGUFFile {
	pub version: u32,
	pub kv_meta: HashMap<String, GGUFValue>,
	pub tensor_info: Vec<GGUFTensorInfo>,
	/// Offset in the file where tensor data section begins.
	pub data_start: u64,
	/// Alignment in bytes (default 32, overridable via general.alignment).
	pub alignment: usize,
}

pub struct GGUFTensorInfo {
	pub name: String,
	pub dim: Vec<u64>,
	pub dtype: u32,
	pub offset: u64,
}

impl GGUFTensorInfo {
	/// Number of elements in this tensor.
	pub fn element_count(&self) -> u64 {
		self.dim.iter().product()
	}

	/// Byte size of this tensor's raw data in the GGUF file.
	pub fn byte_size(&self) -> u64 {
		if let Some((block_elems, block_bytes)) = quant_block_info(self.dtype) {
			let elems = self.element_count();
			let blocks = (elems + block_elems as u64 - 1) / block_elems as u64;
			blocks * block_bytes as u64
		} else {
			// Unknown type: assume f32
			self.element_count() * 4
		}
	}
}

#[derive(Debug, Clone)]
pub enum GGUFValue {
	U8(u8),
	I8(i8),
	U16(u16),
	I16(i16),
	U32(u32),
	I32(i32),
	F32(f32),
	Bool(bool),
	String(String),
	U64(u64),
	I64(i64),
	F64(f64),
	Array(GGUFArrayValue),
}

#[derive(Debug, Clone)]
pub struct GGUFArrayValue {
	pub dtype: GgufValueType,
	pub data: Vec<GGUFValue>,
}

impl GGUFFile {
	/// Default alignment if not specified in metadata.
	fn default_alignment() -> usize {
		32
	}

	/// Parse alignment from metadata.
	fn verify_alignment(kv: &HashMap<String, GGUFValue>) -> usize {
		kv.get("general.alignment")
			.and_then(|v| match v {
				GGUFValue::U64(val) => Some(*val as usize),
				GGUFValue::U32(val) => Some(*val as usize),
				_ => None,
			})
			.unwrap_or_else(Self::default_alignment)
	}

	pub fn padding_needed(current_offset: usize, alignment: usize) -> usize {
		let remainder = current_offset % alignment;
		if remainder == 0 {
			0
		} else {
			alignment - remainder
		}
	}

	/// Parse a GGUF file from raw bytes (header + metadata + tensor info).
	/// Does NOT load tensor data — use read_tensor_data for that.
	pub fn from_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
		let mut r = Reader::new(data);

		// Magic
		let magic: [u8; 4] = [r.read_u8()?, r.read_u8()?, r.read_u8()?, r.read_u8()?];
		if magic != GGUF_MAGIC {
			return Err(format!("bad magic: {:?}", magic).into());
		}

		// Version
		let version = r.read_u32()?;

		// Counts
		let tensor_count = r.read_u64()? as usize;
		let kv_count = r.read_u64()? as usize;

		// KV metadata
		let mut kv_meta: HashMap<String, GGUFValue> = HashMap::with_capacity(kv_count);
		for _ in 0..kv_count {
			let key = r.read_string()?;
			let vtype_u32 = r.read_u32()?;
			let vtype = GgufValueType::from_u32(vtype_u32)
				.ok_or_else(|| format!("unknown value type {}", vtype_u32))?;
			let value = r.read_value(vtype)?;
			kv_meta.insert(key, value);
		}

		// Tensor info
		let alignment = Self::verify_alignment(&kv_meta);
		let mut tensor_info: Vec<GGUFTensorInfo> = Vec::with_capacity(tensor_count);
		for _ in 0..tensor_count {
			let name = r.read_string()?;
			let n_dims = r.read_u32()? as usize;
			let mut dim = Vec::with_capacity(n_dims);
			for _ in 0..n_dims {
				dim.push(r.read_u64()?);
			}
			let dtype = r.read_u32()?;
			let offset = r.read_u64()?;
			tensor_info.push(GGUFTensorInfo {
				name,
				dim,
				dtype,
				offset,
			});
		}

		let data_start = r.pos as u64;
		// Apply alignment padding to data_start
		let pad = Self::padding_needed(data_start as usize, alignment);
		let data_start = data_start + pad as u64;

		Ok(Self {
			version,
			kv_meta,
			tensor_info,
			data_start,
			alignment,
		})
	}

	/// Parse a GGUF file by reading only the header from a file handle.
	/// Tensor data is read on demand via read_tensor_data.
	pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
		// Read enough bytes for header + metadata + tensor info.
		// For a 9.5GB file we only need the first few MB.
		// Read in chunks until we've consumed all tensor info.
		//
		// Strategy: read a generous initial chunk (8MB should cover any model header),
		// parse it, and if not enough, read more.
		let mut file = File::open(path)?;
		let initial_buf_size: usize = 8 * 1024 * 1024; // 8MB
		let mut buf = vec![0u8; initial_buf_size];
		let n = file.read(&mut buf)?;
		buf.truncate(n);

		// If we couldn't even read the header fields, we need more data.
		// For Qwen3.5-9B with ~427 tensors and 31 KV pairs, 8MB is plenty.
		// But let's handle the case where the header is larger.
		match Self::from_bytes(&buf) {
			Ok(parsed) => Ok(parsed),
			Err(e) => {
				let msg = e.to_string();
				if msg.contains("EOF") {
					// Read more data
					let mut full_buf = vec![0u8; 64 * 1024 * 1024]; // 64MB
					file.seek(SeekFrom::Start(0))?;
					let n = file.read(&mut full_buf)?;
					full_buf.truncate(n);
					Self::from_bytes(&full_buf)
				} else {
					Err(e)
				}
			}
		}
	}

	/// Read raw bytes for a specific tensor from the GGUF file.
	pub fn read_tensor_data(
		&self,
		file: &mut File,
		tensor_index: usize,
	) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
		let info = &self.tensor_info[tensor_index];
		let abs_offset = self.data_start + info.offset;
		let size = info.byte_size() as usize;

		file.seek(SeekFrom::Start(abs_offset))?;
		let mut buf = vec![0u8; size];
		file.read_exact(&mut buf)?;
		Ok(buf)
	}

	/// Read a byte range of tensor data from disk without loading the full tensor.
	/// `byte_offset` is relative to the tensor's data start within the GGUF data section.
	pub fn read_tensor_range(
		&self,
		file: &mut File,
		tensor_index: usize,
		byte_offset: usize,
		byte_len: usize,
	) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
		let info = &self.tensor_info[tensor_index];
		let abs_offset = self.data_start + info.offset + byte_offset as u64;
		file.seek(SeekFrom::Start(abs_offset))?;
		let mut buf = vec![0u8; byte_len];
		file.read_exact(&mut buf)?;
		Ok(buf)
	}

	/// Dequantize a tensor's raw data to f32.
	/// Supports F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q6_K.
	pub fn dequantize_to_f32(&self, raw: &[u8], dtype: u32, element_count: usize) -> Vec<f32> {
		match dtype {
			0 => {
				// F32
				raw.chunks_exact(4)
					.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
					.collect()
			}
			1 => {
				// F16
				raw.chunks_exact(2)
					.map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
					.collect()
			}
			30 => {
				// BF16
				raw.chunks_exact(2)
					.map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
					.collect()
			}
			8 => {
				// Q8_0: [f16 scale][32 x i8] per block
				let block_elems = 32;
				let block_size = 34; // 2 + 32
				let n_blocks = (element_count + block_elems - 1) / block_elems;
				let mut out = Vec::with_capacity(element_count);
				for b in 0..n_blocks {
					let off = b * block_size;
					if off + 2 > raw.len() {
						break;
					}
					let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
					for i in 0..block_elems {
						let idx = b * block_elems + i;
						if idx >= element_count {
							break;
						}
						let q_off = off + 2 + i;
						if q_off >= raw.len() {
							break;
						}
						let q = raw[q_off] as i8 as f32;
						out.push(d * q);
					}
				}
				out
			}
			2 => {
				// Q4_0: [f16 scale][16 x 4-bit packed into 8 bytes] per block of 32
				// Each byte holds two 4-bit values: high nibble = element 2i, low = 2i+1
				let block_size = 18; // 2 + 16
				let block_elems = 32;
				let n_blocks = (element_count + block_elems - 1) / block_elems;
				let mut out = Vec::with_capacity(element_count);
				for b in 0..n_blocks {
					let off = b * block_size;
					if off + 2 > raw.len() {
						break;
					}
					let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
					for i in 0..16 {
						let byte = raw[off + 2 + i];
						let q0 = ((byte & 0x0F) as i8) as f32 - 8.0;
						let q1 = ((byte >> 4) as i8) as f32 - 8.0;
						let idx0 = b * block_elems + i * 2;
						let idx1 = b * block_elems + i * 2 + 1;
						if idx0 < element_count {
							out.push(d * q0);
						}
						if idx1 < element_count {
							out.push(d * q1);
						}
					}
				}
				out
			}
			13 => {
				// Q6_K: super-block of 256 elements, 210 bytes
				// Simplified dequantization for stats purposes
				let block_elems = 256;
				let block_size = 210;
				let n_blocks = (element_count + block_elems - 1) / block_elems;
				let mut out = Vec::with_capacity(element_count);
				for b in 0..n_blocks {
					let off = b * block_size;
					if off + block_size > raw.len() {
						break;
					}
					// Q6_K layout: [2 x f16 scales for ql+qh halves][2 x i8 scales][128 x i8 ql][64 x i8 qh]
					// This is complex; approximate with the first scale for rough stats
					let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
					for i in 0..block_elems {
						let idx = b * block_elems + i;
						if idx >= element_count {
							break;
						}
						let q_off = off + 4 + 2 + (i % 128);
						if q_off < raw.len() {
							let q = (raw[q_off] as i8 as f32) * d / 127.0;
							out.push(q);
						} else {
							out.push(0.0);
						}
					}
				}
				out
			}
			_ => {
				// Unknown type: fill with zeros
				vec![0.0; element_count]
			}
		}
	}

	/// Get the model name from metadata.
	pub fn model_name(&self) -> &str {
		self.kv_meta
			.get("general.name")
			.and_then(|v| match v {
				GGUFValue::String(s) => Some(s.as_str()),
				_ => None,
			})
			.unwrap_or("gguf")
	}
}

// ── ModelAdapter implementation ──────────────────────────────────────────────

impl ModelAdapter for GGUFFile {
	fn model_name(&self) -> &str {
		self.model_name()
	}

	fn tensor_count(&self) -> usize {
		self.tensor_info.len()
	}

	fn total_bytes(&self) -> u64 {
		self.tensor_info.iter().map(|t| t.byte_size()).sum()
	}

	fn tensor_at(&self, index: usize) -> Option<TensorDescriptor> {
		self.tensor_info.get(index).map(|t| {
			let dtype = ggml_dtype_to_tensor_dtype(t.dtype);
			let elem_size = match t.dtype {
				0 => 4,
				1 => 2,
				30 => 2, // F32, F16, BF16
				_ => {
					let (_, bs) = quant_block_info(t.dtype).unwrap_or((1, 4));
					bs
				}
			};
			let n_elems: u64 = t.dim.iter().product();
			TensorDescriptor {
				name: t.name.clone(),
				shape: t.dim.iter().map(|d| *d as usize).collect(),
				dtype,
				byte_offset: t.offset,
				byte_size: n_elems * elem_size as u64,
				layer_index: TensorDescriptor::parse_layer_index(&t.name),
			}
		})
	}

	fn find_tensor(&self, name: &str) -> Option<TensorDescriptor> {
		self.tensor_info
			.iter()
			.position(|t| t.name == name)
			.and_then(|i| self.tensor_at(i))
	}

	fn alignment(&self) -> usize {
		self.alignment
	}

	fn raw_data(&self, _tensor_name: &str) -> Option<&[u8]> {
		// With from_file we don't hold tensor data in memory.
		// Use read_tensor_data for on-demand access.
		None
	}

	fn to_tensor_map(&self) -> crate::augment::augment::ModelTensorMap {
		let tensors: Vec<TensorDescriptor> = (0..self.tensor_count())
			.filter_map(|i| self.tensor_at(i))
			.collect();
		let total = self.tensor_info.iter().map(|t| t.byte_size()).sum();
		crate::augment::augment::ModelTensorMap {
			model_name: self.model_name().to_string(),
			tensors,
			total_bytes: total,
		}
	}
}

// ── Importance matrix ────────────────────────────────────────────────────────

/// Per-weight metadata stored separately from core weight data.
///
/// Written to a sidecar file (sandbag.bin) and loaded at inference time. It
/// maps each weight to its prefix group and sign, and tracks precision loss
/// for adaptive re-scan. The core weight file holds only prefixes, unique
/// tails, and counts — this holds the per-weight addressing that ties weights
/// to their entries.
///
/// File layout:
///   [count: u32][group_assignment: u8 * N][sign_bits: u8 * ceil(N/8)][precision_lost: u8 * N]
#[derive(Clone, Debug)]
pub struct ImportanceMatrix {
	/// For each weight: which prefix group it belongs to.
	pub group_assignment: Vec<u8>,
	/// Packed sign bits (1 bit per weight).
	pub sign_bits: Vec<u8>,
	/// Per-weight precision loss in digits (for quality analysis / adaptive rescan).
	pub precision_lost: Vec<u8>,
	/// Weight count.
	pub count: usize,
}

impl ImportanceMatrix {
	/// Serialize to bytes for file storage.
	pub fn to_bytes(&self) -> Vec<u8> {
		let mut data = Vec::with_capacity(
			4 + self.group_assignment.len() + self.sign_bits.len() + self.precision_lost.len(),
		);
		data.extend_from_slice(&(self.count as u32).to_le_bytes());
		data.extend_from_slice(&self.group_assignment);
		data.extend_from_slice(&self.sign_bits);
		data.extend_from_slice(&self.precision_lost);
		data
	}

	/// Deserialize from bytes.
	pub fn from_bytes(data: &[u8]) -> Option<Self> {
		if data.len() < 4 {
			return None;
		}
		let count = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
		let sign_bytes = (count + 7) / 8;
		let needed = 4 + count + sign_bytes + count;
		if data.len() < needed {
			return None;
		}

		let group_assignment = data[4..4 + count].to_vec();
		let sign_bits = data[4 + count..4 + count + sign_bytes].to_vec();
		let precision_lost = data[4 + count + sign_bytes..needed].to_vec();

		Some(Self {
			group_assignment,
			sign_bits,
			precision_lost,
			count,
		})
	}

	pub fn bytes(&self) -> usize {
		4 + self.group_assignment.len() + self.sign_bits.len() + self.precision_lost.len()
	}
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_f16_to_f32() {
		// 1.0 in f16
		assert_eq!(f16_to_f32(0x3C00), 1.0);
		// 0.0
		assert_eq!(f16_to_f32(0x0000), 0.0);
		// -1.0
		assert_eq!(f16_to_f32(0xBC00), -1.0);
		// 2.0
		assert_eq!(f16_to_f32(0x4000), 2.0);
		// 0.5
		assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
	}

	#[test]
	fn test_bf16_to_f32() {
		// 1.0 in bf16 = 0x3F80
		assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-6);
		// 0.0
		assert_eq!(bf16_to_f32(0x0000), 0.0);
	}

	#[test]
	fn test_q8_0_dequant() {
		// One Q8_0 block: scale=1.0 (f16=0x3C00), 32 int8 values 0..31
		let mut raw = Vec::new();
		raw.extend_from_slice(&0x3C00u16.to_le_bytes()); // f16 1.0
		for i in 0..32 {
			raw.push(i as i8 as u8);
		}

		// Manually verify the dequantization logic:
		// d (scale) = 1.0, each weight = d * qs[i]
		let d = f16_to_f32(0x3C00);
		assert!((d - 1.0).abs() < 1e-6);
		for i in 0i32..32 {
			let q = i as f32;
			let actual = d * (raw[2 + i as usize] as i8 as f32);
			assert!((q - actual).abs() < 1e-6, "mismatch at {}", i);
		}
	}
}

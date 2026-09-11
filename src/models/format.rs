#![allow(nonstandard_style)]

//! Format structs: safetensors, gguf, pickletensor, sandbag.
//!
//! Sandbag layout (sequential, GPU-ready):
//!   [HEADER]
//!     u64  magic          = 0x5342_5342 ("SBSB")
//!     u32  version        = 1
//!     u64  num_tensors
//!     u64  total_data_bytes
//!
//!   [TENSOR_INDEX]  — one entry per tensor, in order
//!     u64  name_len
//!     [u8; name_len]  name_bytes  (UTF-8)
//!     u64  num_dims
//!     [u64; num_dims]  shape
//!     u8   quant_scheme   (see GgmlType)
//!     u64  data_offset    — byte offset from start of DATA section
//!     u64  data_len       — compressed byte length in DATA section
//!     u16  prefix          — 2-digit prefix (0..99), fits in u16
//!     u64  sign_bits        — sign bit block (one bit per element, packed as u64 words)
//!
//!   [DATA]          — sequential compressed payload, tensors back-to-back
//!     Each tensor's data is written exactly where data_offset says.
//!     Decompression: value = (prefix * 100 + tail) * sign
//!       prefix  = 2-digit u8  (0..99)
//!       tail    = 3-digit reconstructed from palette index (0..255) + RLE tokens (256..999)
//!       sign    = bit from sign_bits block

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ggml_type — mirrors ggml.h enum exactly (from llama.cpp)
// Every quantization type that GGUF files can actually contain.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum GgmlType {
	F32 = 0,
	F16 = 1,
	Q4_0 = 2,
	Q4_1 = 3,
	// Q4_2 = 4, removed
	// Q4_3 = 5, removed
	Q5_0 = 6,
	Q5_1 = 7,
	Q8_0 = 8,
	Q8_1 = 9,
	Q2_K = 10,
	Q3_K = 11,
	Q4_K = 12,
	Q5_K = 13,
	Q6_K = 14,
	Q8_K = 15,
	IQ2_XXS = 16,
	IQ2_XS = 17,
	IQ3_XXS = 18,
	IQ1_S = 19,
	IQ4_NL = 20,
	IQ3_S = 21,
	IQ2_S = 22,
	IQ4_XS = 23,
	I8 = 24,
	I16 = 25,
	I32 = 26,
	I64 = 27,
	F64 = 28,
	IQ1_M = 29,
	BF16 = 30,
	// Q4_0_4_4 = 31, removed
	// Q4_0_4_8 = 32, removed
	// Q4_0_8_8 = 33, removed
	TQ1_0 = 34,
	TQ2_0 = 35,
	// IQ4_NL_4_4 = 36, removed
	// IQ4_NL_4_8 = 37, removed
	// IQ4_NL_8_8 = 38, removed
	MXFP4 = 39,
	NVFP4 = 40,
	Q1_0 = 41,
	Q2_0 = 42,
	/// Proprietary sandbag: prefix/tail/sign-bit layout.
	Sandbag = 127,
}

impl GgmlType {
	pub fn name(self) -> &'static str {
		match self {
			GgmlType::F32 => "F32",
			GgmlType::F16 => "F16",
			GgmlType::Q4_0 => "Q4_0",
			GgmlType::Q4_1 => "Q4_1",
			GgmlType::Q5_0 => "Q5_0",
			GgmlType::Q5_1 => "Q5_1",
			GgmlType::Q8_0 => "Q8_0",
			GgmlType::Q8_1 => "Q8_1",
			GgmlType::Q2_K => "Q2_K",
			GgmlType::Q3_K => "Q3_K",
			GgmlType::Q4_K => "Q4_K",
			GgmlType::Q5_K => "Q5_K",
			GgmlType::Q6_K => "Q6_K",
			GgmlType::Q8_K => "Q8_K",
			GgmlType::IQ2_XXS => "IQ2_XXS",
			GgmlType::IQ2_XS => "IQ2_XS",
			GgmlType::IQ3_XXS => "IQ3_XXS",
			GgmlType::IQ1_S => "IQ1_S",
			GgmlType::IQ4_NL => "IQ4_NL",
			GgmlType::IQ3_S => "IQ3_S",
			GgmlType::IQ2_S => "IQ2_S",
			GgmlType::IQ4_XS => "IQ4_XS",
			GgmlType::I8 => "I8",
			GgmlType::I16 => "I16",
			GgmlType::I32 => "I32",
			GgmlType::I64 => "I64",
			GgmlType::F64 => "F64",
			GgmlType::IQ1_M => "IQ1_M",
			GgmlType::BF16 => "BF16",
			GgmlType::TQ1_0 => "TQ1_0",
			GgmlType::TQ2_0 => "TQ2_0",
			GgmlType::MXFP4 => "MXFP4",
			GgmlType::NVFP4 => "NVFP4",
			GgmlType::Q1_0 => "Q1_0",
			GgmlType::Q2_0 => "Q2_0",
			GgmlType::Sandbag => "Sandbag",
		}
	}

	pub fn is_float(self) -> bool {
		matches!(
			self,
			GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::F64
		)
	}

	pub fn is_integer(self) -> bool {
		matches!(
			self,
			GgmlType::I8 | GgmlType::I16 | GgmlType::I32 | GgmlType::I64
		)
	}

	pub fn is_quantized(self) -> bool {
		!self.is_float() && !self.is_integer() && self != GgmlType::Sandbag
	}
}

impl std::fmt::Display for GgmlType {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.name())
	}
}

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------
/// Parsed safetensors header (JSON at the top of the file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorsHeader {
	#[serde(rename = "__metadata__", default)]
	pub metadata: HashMap<String, String>,
	#[serde(flatten)]
	pub tensors: HashMap<String, SafeTensorInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorInfo {
	pub dtype: String,
	pub shape: Vec<u64>,
	pub data_offsets: (u64, u64), // (start, end)
}

/// Layer kind returned by get_layer_type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
	Attention,
	Activation,   // Wide upward expansions (up_proj, gate_proj)
	Contraction,  // Wide downward contractions (down_proj) - crucial for tile flipping
	Weight,       // General weights/biases
	OutputProj,
	Norm,
	Embedding,
	Unknown,
}

impl std::fmt::Display for LayerType {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			LayerType::Attention => write!(f, "attention"),
			LayerType::Activation => write!(f, "activation"),
			LayerType::Contraction => write!(f, "contraction"),
			LayerType::Weight => write!(f, "weight"),
			LayerType::OutputProj => write!(f, "output_projection"),
			LayerType::Norm => write!(f, "normalization"),
			LayerType::Embedding => write!(f, "embedding"),
			LayerType::Unknown => write!(f, "unknown"),
		}
	}
}

pub fn get_layer_type(name: &str) -> LayerType {
	let lower = name.to_lowercase();
	if lower.contains("self_attn")
		|| lower.contains("attention")
		|| lower.contains("q_proj")
		|| lower.contains("k_proj")
		|| lower.contains("v_proj")
		|| lower.contains("attn")
	{
		return LayerType::Attention;
	}
	// Explicitly catch the inverse down-projection first
	if lower.contains("down_proj") {
		return LayerType::Contraction;
	}
	if lower.contains("gate_proj")
		|| lower.contains("up_proj")
		|| lower.contains("mlp")
	{
		return LayerType::Activation;
	}
	if lower.contains("output") || lower.contains("lm_head") || lower.contains("embed_tokens") {
		return LayerType::OutputProj;
	}
	if lower.contains("norm") {
		return LayerType::Norm;
	}
	if lower.contains("embedding") {
		return LayerType::Embedding;
	}
	if lower.contains("weight") || lower.contains("bias") {
		return LayerType::Weight;
	}
	LayerType::Unknown
}


// ---------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufHeader {
	pub version: u32,
	pub kv_meta: HashMap<String, GgufValue>,
	pub tensor_info: Vec<GgufTensorHeader>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GgufValue {
	Int8(i8),
	Int16(i16),
	Int32(i32),
	Int64(i64),
	Uint8(u8),
	Uint16(u16),
	Uint32(u32),
	Uint64(u64),
	Float32(f32),
	Float64(f64),
	Bool(bool),
	String(String),
	Array(Vec<GgufValue>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTensorHeader {
	pub name: String,
	/// Wire-level GGML type from the tensor-info record. Note this is the tensor's
	/// own type (e.g. Q4_K), not the file's marketing name (Q4_K_M) — those
	/// distinctions live only in filenames, never in the file.
	pub dtype: GgmlType,
	pub shape: Vec<u64>,
	/// Absolute byte offset of this tensor's payload within the file.
	pub data_offset: u64,
	pub alignment_padding: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GgufDtype {
	F32,
	F16,
	BF16,
	Q4_0,
	Q4_1,
	Q5_0,
	Q5_1,
	Q8_0,
	Q8_1,
	Q2_K,
	Q3_K_M,
	Q3_K_S,
	Q3_K_L,
	Q4_K_M,
	Q4_K_S,
	Q4_K_L,
	Q5_K_M,
	Q5_K_S,
	Q5_K_L,
	Q6_K,
	IQ1_M,
	IQ1_S,
	IQ2_XS,
	IQ2_S,
	IQ2_M,
	IQ3_XS,
	IQ3_S,
	IQ3_M,
	IQ4_XS,
	IQ4_S,
	IQ4_M,
	IQ4XL,
}

impl GgufDtype {
	/// Map each GGUF variant to its exact GgmlType — no lossy bucketing.
	pub fn to_ggml_type(&self) -> GgmlType {
		match self {
			GgufDtype::F32 => GgmlType::F32,
			GgufDtype::F16 => GgmlType::F16,
			GgufDtype::BF16 => GgmlType::BF16,
			GgufDtype::Q4_0 => GgmlType::Q4_0,
			GgufDtype::Q4_1 => GgmlType::Q4_1,
			GgufDtype::Q5_0 => GgmlType::Q5_0,
			GgufDtype::Q5_1 => GgmlType::Q5_1,
			GgufDtype::Q8_0 => GgmlType::Q8_0,
			GgufDtype::Q8_1 => GgmlType::Q8_1,
			GgufDtype::Q2_K => GgmlType::Q2_K,
			GgufDtype::Q3_K_M | GgufDtype::Q3_K_S | GgufDtype::Q3_K_L => GgmlType::Q3_K,
			GgufDtype::Q4_K_M | GgufDtype::Q4_K_S | GgufDtype::Q4_K_L => GgmlType::Q4_K,
			GgufDtype::Q5_K_M | GgufDtype::Q5_K_S | GgufDtype::Q5_K_L => GgmlType::Q5_K,
			GgufDtype::Q6_K => GgmlType::Q6_K,
			GgufDtype::IQ1_M => GgmlType::IQ1_M,
			GgufDtype::IQ1_S => GgmlType::IQ1_S,
			GgufDtype::IQ2_XS => GgmlType::IQ2_XS,
			GgufDtype::IQ2_S => GgmlType::IQ2_S,
			GgufDtype::IQ2_M => GgmlType::IQ2_S, // IQ2_M → IQ2_S (closest)
			GgufDtype::IQ3_XS => GgmlType::IQ3_XXS,
			GgufDtype::IQ3_S => GgmlType::IQ3_S,
			GgufDtype::IQ3_M => GgmlType::IQ3_S,
			GgufDtype::IQ4_XS => GgmlType::IQ4_XS,
			GgufDtype::IQ4_S => GgmlType::IQ4_NL,
			GgufDtype::IQ4_M => GgmlType::IQ4_NL,
			GgufDtype::IQ4XL => GgmlType::IQ4_XS,
		}
	}
}

impl GgmlType {
	/// Decode a GGML type id as written in a GGUF tensor-info record.
	pub fn from_u32(v: u32) -> Option<Self> {
		use GgmlType::*;
		Some(match v {
			0 => F32,
			1 => F16,
			2 => Q4_0,
			3 => Q4_1,
			6 => Q5_0,
			7 => Q5_1,
			8 => Q8_0,
			9 => Q8_1,
			10 => Q2_K,
			11 => Q3_K,
			12 => Q4_K,
			13 => Q5_K,
			14 => Q6_K,
			15 => Q8_K,
			16 => IQ2_XXS,
			17 => IQ2_XS,
			18 => IQ3_XXS,
			19 => IQ1_S,
			20 => IQ4_NL,
			21 => IQ3_S,
			22 => IQ2_S,
			23 => IQ4_XS,
			24 => I8,
			25 => I16,
			26 => I32,
			27 => I64,
			28 => F64,
			29 => IQ1_M,
			30 => BF16,
			34 => TQ1_0,
			35 => TQ2_0,
			39 => MXFP4,
			40 => NVFP4,
			41 => Q1_0,
			42 => Q2_0,
			127 => Sandbag,
			_ => return None,
		})
	}

	/// Elements per stored block. 1 for unquantized types.
	pub fn block_size(self) -> u64 {
		use GgmlType::*;
		match self {
			F32 | F16 | BF16 | F64 | I8 | I16 | I32 | I64 => 1,
			Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | IQ4_NL | MXFP4 | NVFP4 => 32,
			Sandbag => 1,
			// Every K- and IQ-quant is a 256-element superblock.
			_ => 256,
		}
	}

	/// Bytes per stored block.
	pub fn type_size(self) -> u64 {
		use GgmlType::*;
		match self {
			F32 | I32 => 4,
			F16 | BF16 | I16 => 2,
			F64 | I64 => 8,
			I8 => 1,
			Q4_0 => 18,
			Q4_1 => 20,
			Q5_0 => 22,
			Q5_1 => 24,
			Q8_0 => 34,
			Q8_1 => 40,
			Q2_K => 84,
			Q3_K => 110,
			Q4_K => 144,
			Q5_K => 176,
			Q6_K => 210,
			Q8_K => 292,
			IQ2_XXS => 66,
			IQ2_XS => 74,
			IQ2_S => 82,
			IQ3_XXS => 98,
			IQ3_S => 110,
			IQ1_S => 50,
			IQ1_M => 56,
			IQ4_NL => 18,
			IQ4_XS => 136,
			TQ1_0 => 54,
			TQ2_0 => 66,
			MXFP4 => 17,
			NVFP4 => 18,
			Q1_0 => 12,
			Q2_0 => 20,
			// Sandbag: prefix + tail per element, sign plane accounted separately.
			Sandbag => 2,
		}
	}

	/// Stored byte length of a tensor with `elem_count` elements.
	pub fn tensor_bytes(self, elem_count: u64) -> u64 {
		let bs = self.block_size();
		elem_count.div_ceil(bs) * self.type_size()
	}
}

/// Little-endian cursor over a GGUF file.
struct GgufCursor<'a> {
	data: &'a [u8],
	pos: usize,
}

impl<'a> GgufCursor<'a> {
	fn new(data: &'a [u8]) -> Self {
		Self { data, pos: 0 }
	}

	fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
		let end = self
			.pos
			.checked_add(n)
			.ok_or_else(|| "GGUF cursor overflow".to_string())?;
		let s = self
			.data
			.get(self.pos..end)
			.ok_or_else(|| format!("GGUF truncated: wanted {} bytes at {}", n, self.pos))?;
		self.pos = end;
		Ok(s)
	}

	fn u8(&mut self) -> Result<u8, String> {
		Ok(self.take(1)?[0])
	}
	fn u16(&mut self) -> Result<u16, String> {
		Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
	}
	fn u32(&mut self) -> Result<u32, String> {
		Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
	}
	fn u64(&mut self) -> Result<u64, String> {
		Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
	}
	fn f32(&mut self) -> Result<f32, String> {
		Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
	}
	fn f64(&mut self) -> Result<f64, String> {
		Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
	}

	/// GGUF string: u64 length then raw UTF-8.
	fn string(&mut self) -> Result<String, String> {
		let len = self.u64()? as usize;
		let bytes = self.take(len)?;
		String::from_utf8(bytes.to_vec()).map_err(|e| format!("GGUF string not UTF-8: {}", e))
	}

	/// One metadata value of the given wire type.
	fn value(&mut self, ty: u32) -> Result<GgufValue, String> {
		Ok(match ty {
			0 => GgufValue::Uint8(self.u8()?),
			1 => GgufValue::Int8(self.u8()? as i8),
			2 => GgufValue::Uint16(self.u16()?),
			3 => GgufValue::Int16(self.u16()? as i16),
			4 => GgufValue::Uint32(self.u32()?),
			5 => GgufValue::Int32(self.u32()? as i32),
			6 => GgufValue::Float32(self.f32()?),
			7 => GgufValue::Bool(self.u8()? != 0),
			8 => GgufValue::String(self.string()?),
			9 => {
				let elem_ty = self.u32()?;
				let count = self.u64()? as usize;
				let mut items = Vec::with_capacity(count.min(1 << 20));
				for _ in 0..count {
					items.push(self.value(elem_ty)?);
				}
				GgufValue::Array(items)
			}
			10 => GgufValue::Uint64(self.u64()?),
			11 => GgufValue::Int64(self.u64()? as i64),
			12 => GgufValue::Float64(self.f64()?),
			other => return Err(format!("Unknown GGUF value type {}", other)),
		})
	}
}

/// Parse a GGUF file's header, metadata and tensor index.
///
/// `data` must cover at least the header; tensor payloads are not touched.
/// `data_offset` on each returned tensor is absolute within the file, so callers
/// can slice weights directly without re-deriving the alignment.
pub fn parse_gguf(data: &[u8]) -> Result<GgufHeader, String> {
	let mut c = GgufCursor::new(data);

	let magic = c.take(4)?;
	if magic != b"GGUF" {
		return Err(format!("Not a GGUF file: magic {:?}", magic));
	}
	let version = c.u32()?;
	if version != 2 && version != 3 {
		return Err(format!("Unsupported GGUF version {}", version));
	}

	let tensor_count = c.u64()?;
	let kv_count = c.u64()?;

	let mut kv_meta = HashMap::with_capacity(kv_count as usize);
	for _ in 0..kv_count {
		let key = c.string()?;
		let ty = c.u32()?;
		let val = c.value(ty)?;
		kv_meta.insert(key, val);
	}

	// Tensor data is aligned; the alignment is itself a metadata key.
	let alignment = match kv_meta.get("general.alignment") {
		Some(GgufValue::Uint32(v)) => *v as u64,
		Some(GgufValue::Uint64(v)) => *v,
		_ => 32,
	};

	// Tensor-info records come first, then padding, then the payloads.
	let mut raw_infos = Vec::with_capacity(tensor_count as usize);
	for i in 0..tensor_count {
		let name = c.string()?;
		let n_dims = c.u32()? as usize;
		let mut shape = Vec::with_capacity(n_dims);
		for _ in 0..n_dims {
			shape.push(c.u64()?);
		}
		let type_id = c.u32()?;
		let dtype = GgmlType::from_u32(type_id)
			.ok_or_else(|| format!("Tensor {} ({}) has unknown GGML type {}", i, name, type_id))?;
		let rel_offset = c.u64()?;
		raw_infos.push((name, shape, dtype, rel_offset));
	}

	let data_start = c.pos.next_multiple_of(alignment as usize) as u64;

	let tensor_info = raw_infos
		.into_iter()
		.map(|(name, shape, dtype, rel_offset)| GgufTensorHeader {
			name,
			dtype,
			shape,
			data_offset: data_start + rel_offset,
			alignment_padding: 0,
		})
		.collect();

	Ok(GgufHeader {
		version,
		kv_meta,
		tensor_info,
	})
}

pub fn gguf_get_layer_dim(header: &GgufHeader, layer_name: &str) -> Option<Vec<u64>> {
	header
		.tensor_info
		.iter()
		.find(|t| t.name == layer_name)
		.map(|t| t.shape.clone())
}

pub fn gguf_get_layer_type(name: &str) -> LayerType {
	get_layer_type(name)
}

// ---------------------------------------------------------------------------
// PickleTensor (PyTorch .pt / .pth)
// ---------------------------------------------------------------------------
/// Minimal pickle tensor descriptor — full deserialization lives in the loader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PickleTensorInfo {
	pub name: String,
	pub shape: Vec<u64>,
	pub dtype: String, // e.g. "Float", "Half", "BFloat16"
	pub data_offset: u64,
	pub data_len: u64,
}

// ---------------------------------------------------------------------------
// Sandbag (proprietary sequential quantized format)
// ---------------------------------------------------------------------------
/// Magic bytes: 0x5342_5342 ("SBSB").
pub const SANDBAG_MAGIC: u64 = 0x5342_5342;
pub const SANDBAG_VERSION: u32 = 1;
/// magic u64 + version u32 + num_tensors u64 + total_data_bytes u64.
pub const SANDBAG_HEADER_BYTES: usize = 8 + 4 + 8 + 8;

/// Weights per scale block. Fixed at 32 so one block's sign bits are exactly one
/// u32, which is what lets a single GPU invocation own a block outright.
pub const SANDBAG_BLOCK_ELEMS: u64 = 32;

/// Payload size of one tensor. Three planes, in this order:
///
/// ```text
///   [ scales ]  ceil(N/32) x f16   per-block scale
///   [ pairs  ]  N x (prefix_u8, tail_u8)
///   [ signs  ]  ceil(N/64) x u64   at the very end, per spec
/// ```
///
/// Fixed-rate by construction — the size depends only on `elem_count`, never on
/// the values. That is the property that makes offsets derivable and lets the
/// shader write without a prefix-sum or any stored offset table.
pub fn sandbag_tensor_bytes(elem_count: u64) -> u64 {
	sandbag_scale_bytes(elem_count) + elem_count * 2 + elem_count.div_ceil(64) * 8
}

/// Size of the per-block scale plane.
pub fn sandbag_scale_bytes(elem_count: u64) -> u64 {
	elem_count.div_ceil(SANDBAG_BLOCK_ELEMS) * 2
}

/// Byte offset of the scale plane within a tensor's payload. It leads, so this
/// is zero — named rather than inlined so the three planes read symmetrically.
pub fn sandbag_scale_offset_in_tensor() -> u64 {
	0
}

/// Byte offset of the prefix/tail plane within a tensor's payload.
pub fn sandbag_pairs_offset(elem_count: u64) -> u64 {
	sandbag_scale_bytes(elem_count)
}

/// Byte offset of the sign plane within a tensor's payload.
pub fn sandbag_sign_offset(elem_count: u64) -> u64 {
	sandbag_scale_bytes(elem_count) + elem_count * 2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandbagHeader {
	pub magic: u64,
	pub version: u32,
	pub num_tensors: u64,
	pub total_data_bytes: u64,
}

impl SandbagHeader {
	pub fn new(num_tensors: u64, total_data_bytes: u64) -> Self {
		Self {
			magic: SANDBAG_MAGIC,
			version: SANDBAG_VERSION,
			num_tensors,
			total_data_bytes,
		}
	}
}

/// Single tensor entry in the sandbag index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandbagTensorEntry {
	pub name: String,
	pub shape: Vec<u64>,
	pub quant_scheme: u8,
	pub data_offset: u64,    // offset within DATA section
	pub data_len: u64,       // compressed byte length
	pub prefix: u16,         // 2-digit prefix (0..99)
	pub sign_bits: Vec<u64>, // sign bit block, packed as u64 words
}

impl SandbagTensorEntry {
	pub fn new(
		name: String,
		shape: Vec<u64>,
		quant_scheme: GgmlType,
		data_offset: u64,
		data_len: u64,
		prefix: u16,
		sign_bits: Vec<u64>,
	) -> Self {
		Self {
			name,
			shape,
			quant_scheme: quant_scheme as u8,
			data_offset,
			data_len,
			prefix,
			sign_bits,
		}
	}
}

/// Validate a sandbag header from raw bytes.
pub fn validate_sandbag_header(data: &[u8]) -> Result<SandbagHeader, String> {
	if data.len() < SANDBAG_HEADER_BYTES {
		return Err("Sandbag file too small for header".into());
	}
	let magic = u64::from_le_bytes(data[0..8].try_into().unwrap());
	if magic != SANDBAG_MAGIC {
		return Err(format!("Invalid sandbag magic: 0x{:016X}", magic));
	}
	let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
	let num_tensors = u64::from_le_bytes(data[12..20].try_into().unwrap());
	// 20..28, not 20..24 — this is a u64 and the header is 28 bytes, not 24.
	let total_data_bytes = u64::from_le_bytes(data[20..28].try_into().unwrap());
	Ok(SandbagHeader {
		magic: SANDBAG_MAGIC,
		version,
		num_tensors,
		total_data_bytes,
	})
}

/// Sandbag file reader — sequential read, no rearranging.
pub struct SandbagReader {
	data: Vec<u8>,
	header: SandbagHeader,
	tensor_entries: Vec<SandbagTensorEntry>,
	/// Byte offset where the DATA section begins (end of the index).
	data_start: usize,
}

impl SandbagReader {
	pub fn from_path(path: &std::path::Path) -> Result<Self, String> {
		let data = std::fs::read(path).map_err(|e| format!("Read failed: {}", e))?;
		Self::from_bytes(&data)
	}

	/// Parse header + index. Data offsets are *derived*, not stored: the payload
	/// is fixed-rate, so each tensor's size follows from its shape alone and the
	/// offsets are a running sum. That is what lets the writer stream straight out
	/// of the GPU without recording an offset table.
	pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
		let header = validate_sandbag_header(data)?;

		let mut entries = Vec::with_capacity(header.num_tensors as usize);
		let mut cursor = SANDBAG_HEADER_BYTES;
		let mut data_offset: u64 = 0;

		for i in 0..header.num_tensors {
			let need = |cursor: usize, n: usize| -> Result<(), String> {
				if cursor + n > data.len() {
					Err(format!("Truncated sandbag index at tensor {}", i))
				} else {
					Ok(())
				}
			};

			need(cursor, 8)?;
			let name_len =
				u64::from_le_bytes(data[cursor..cursor + 8].try_into().unwrap()) as usize;
			cursor += 8;

			need(cursor, name_len)?;
			let name = String::from_utf8(data[cursor..cursor + name_len].to_vec())
				.map_err(|e| format!("Tensor {} name is not UTF-8: {}", i, e))?;
			cursor += name_len;

			need(cursor, 8)?;
			let n_dims = u64::from_le_bytes(data[cursor..cursor + 8].try_into().unwrap()) as usize;
			cursor += 8;

			need(cursor, n_dims * 8)?;
			let mut shape = Vec::with_capacity(n_dims);
			for d in 0..n_dims {
				let off = cursor + d * 8;
				shape.push(u64::from_le_bytes(data[off..off + 8].try_into().unwrap()));
			}
			cursor += n_dims * 8;

			need(cursor, 1)?;
			let quant_scheme = data[cursor];
			cursor += 1;

			let elem_count: u64 = shape.iter().product();
			let data_len = sandbag_tensor_bytes(elem_count);

			entries.push(SandbagTensorEntry {
				name,
				shape,
				quant_scheme,
				data_offset,
				data_len,
				prefix: 0,
				sign_bits: Vec::new(),
			});
			data_offset += data_len;
		}

		Ok(Self {
			data: data.to_vec(),
			header,
			tensor_entries: entries,
			data_start: cursor,
		})
	}

	pub fn header(&self) -> &SandbagHeader {
		&self.header
	}

	pub fn entries(&self) -> &[SandbagTensorEntry] {
		&self.tensor_entries
	}

	/// Tensor names in file order — order is the format's load-bearing invariant.
	pub fn tensor_names(&self) -> Vec<&str> {
		self.tensor_entries
			.iter()
			.map(|e| e.name.as_str())
			.collect()
	}

	pub fn entry(&self, name: &str) -> Option<&SandbagTensorEntry> {
		self.tensor_entries.iter().find(|e| e.name == name)
	}

	/// Raw quantized bytes for one tensor: prefix/tail pairs followed by the
	/// sign-bit u64 words. Returns `None` if the file is short.
	pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
		let e = self.entry(name)?;
		let start = self.data_start + e.data_offset as usize;
		let end = start + e.data_len as usize;
		self.data.get(start..end)
	}
}

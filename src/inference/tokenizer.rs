// BPE tokenizer for Qwen3.5 — extracts vocabulary and merges from GGUF metadata.
//
// Implements byte-level BPE with the GPT-2 byte-to-unicode mapping:
//   1. Pretokenize text using the GPT-2 regex pattern
//   2. Convert each piece to UTF-8 bytes, then map each byte to a unicode char
//   3. Apply BPE merges to each piece
//   4. Look up the resulting tokens in the vocab
//
// Decode reverses the process: token ID → vocab string → reverse byte mapping
// → UTF-8 decode.

use crate::models::formats::gguf::{GGUFFile, GGUFValue};
use std::collections::HashMap;

/// Serializable raw tokenizer data for storing to disk.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TokenizerData {
	#[serde(default)]
	pub model: serde_json::Value,
	#[serde(default)]
	pub pre: serde_json::Value,
	#[serde(default)]
	pub tokens: Vec<String>,
	#[serde(default)]
	pub token_types: Vec<u32>,
	#[serde(default)]
	pub merges: Vec<String>,
	#[serde(default)]
	pub eos_token_id: u32,
	#[serde(default)]
	pub pad_token_id: u32,
	#[serde(default)]
	pub chat_template: String,
	#[serde(default)]
	pub added_tokens: Vec<AddedToken>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct AddedToken {
	#[serde(default)]
	pub id: u32,
	#[serde(default)]
	pub content: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Tokenizer {
	pub vocab: Vec<String>,
	pub token_type: Vec<u32>,
	pub eos_token_id: u32,
	pub pad_token_id: u32,
	/// BPE merge ranks: (first_token_id, second_token_id) → merge_rank.
	/// Lower rank = higher priority (applied first).
	merge_ranks: HashMap<(u32, u32), u32>,
	/// Token ID lookup: token string → token ID.
	token_to_id: HashMap<String, u32>,
	/// Byte-to-unicode mapping (GPT-2 byte-level encoding).
	byte_to_unicode: Vec<char>,
	/// Reverse mapping: unicode char → byte.
	unicode_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
	/// Build a Tokenizer from GGUF metadata.
	pub fn from_gguf(gguf: &GGUFFile) -> Self {
		// Extract a string array from GGUF KV metadata.
		let get_string_array = |key: &str| -> Vec<String> {
			gguf.kv_meta
				.get(key)
				.and_then(|v| match v {
					GGUFValue::Array(arr) => Some(
						arr.data
							.iter()
							.filter_map(|item| {
								if let GGUFValue::String(s) = item {
									Some(s.clone())
								} else {
									None
								}
							})
							.collect(),
					),
					_ => None,
				})
				.unwrap_or_default()
		};

		// Extract a u32 array from GGUF KV metadata.
		let get_u32_array = |key: &str| -> Vec<u32> {
			gguf.kv_meta
				.get(key)
				.and_then(|v| match v {
					GGUFValue::Array(arr) => Some(
						arr.data
							.iter()
							.filter_map(|item| match item {
								GGUFValue::U32(v) => Some(*v),
								GGUFValue::U64(v) => Some(*v as u32),
								_ => None,
							})
							.collect(),
					),
					_ => None,
				})
				.unwrap_or_default()
		};

		// Extract a u32 scalar from GGUF KV metadata.
		let get_u32 = |key: &str| -> u32 {
			gguf.kv_meta
				.get(key)
				.and_then(|v| match v {
					GGUFValue::U32(v) => Some(*v),
					GGUFValue::U64(v) => Some(*v as u32),
					_ => None,
				})
				.unwrap_or(0)
		};

		let vocab = get_string_array("tokenizer.ggml.tokens");
		let token_type = get_u32_array("tokenizer.ggml.token_type");
		let merges = get_string_array("tokenizer.ggml.merges");
		let eos_token_id = get_u32("tokenizer.ggml.eos_token_id");
		let pad_token_id = get_u32("tokenizer.ggml.padding_token_id");

		// Build token_to_id: token string → token ID.
		let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(vocab.len());
		for (id, token) in vocab.iter().enumerate() {
			token_to_id.insert(token.clone(), id as u32);
		}

		// Build merge_ranks: (first_token_id, second_token_id) → rank.
		let mut merge_ranks: HashMap<(u32, u32), u32> = HashMap::with_capacity(merges.len());
		for (rank, merge) in merges.iter().enumerate() {
			if let Some(space_idx) = merge.find(' ') {
				let first = &merge[..space_idx];
				let second = &merge[space_idx + 1..];
				if let (Some(&id1), Some(&id2)) = (token_to_id.get(first), token_to_id.get(second))
				{
					merge_ranks.insert((id1, id2), rank as u32);
				}
			}
		}

		// Build the GPT-2 byte-to-unicode mapping.
		let (byte_to_unicode, unicode_to_byte) = Self::build_byte_mapping();

		Self {
			vocab,
			token_type,
			eos_token_id,
			pad_token_id,
			merge_ranks,
			token_to_id,
			byte_to_unicode,
			unicode_to_byte,
		}
	}

	/// Extract raw tokenizer data from a GGUF file.
	pub fn extract_data(gguf: &GGUFFile) -> TokenizerData {
		let get_string = |key: &str| -> String {
			gguf.kv_meta
				.get(key)
				.and_then(|v| match v {
					GGUFValue::String(s) => Some(s.clone()),
					_ => None,
				})
				.unwrap_or_default()
		};
		let get_u32 = |key: &str| -> u32 {
			gguf.kv_meta
				.get(key)
				.and_then(|v| match v {
					GGUFValue::U32(v) => Some(*v),
					GGUFValue::U64(v) => Some(*v as u32),
					_ => None,
				})
				.unwrap_or(0)
		};
		let get_string_array = |key: &str| -> Vec<String> {
			gguf.kv_meta
				.get(key)
				.and_then(|v| match v {
					GGUFValue::Array(arr) => Some(
						arr.data
							.iter()
							.filter_map(|item| {
								if let GGUFValue::String(s) = item {
									Some(s.clone())
								} else {
									None
								}
							})
							.collect(),
					),
					_ => None,
				})
				.unwrap_or_default()
		};
		let get_u32_array = |key: &str| -> Vec<u32> {
			gguf.kv_meta
				.get(key)
				.and_then(|v| match v {
					GGUFValue::Array(arr) => Some(
						arr.data
							.iter()
							.filter_map(|item| match item {
								GGUFValue::U32(v) => Some(*v),
								GGUFValue::U64(v) => Some(*v as u32),
								_ => None,
							})
							.collect(),
					),
					_ => None,
				})
				.unwrap_or_default()
		};

		TokenizerData {
			model: serde_json::Value::String(get_string("tokenizer.ggml.model")),
			pre: serde_json::Value::String(get_string("tokenizer.ggml.pre")),
			tokens: get_string_array("tokenizer.ggml.tokens"),
			token_types: get_u32_array("tokenizer.ggml.token_type"),
			merges: get_string_array("tokenizer.ggml.merges"),
			eos_token_id: get_u32("tokenizer.ggml.eos_token_id"),
			pad_token_id: get_u32("tokenizer.ggml.padding_token_id"),
			chat_template: get_string("tokenizer.ggml.chat_template"),
			added_tokens: Vec::new(),
		}
	}

	/// Build tokenizer from raw data.
	fn from_data(data: TokenizerData) -> Self {
		let (vocab, merges) = if data.tokens.is_empty() {
			let model = data
				.model
				.as_object()
				.map(|m| m.clone())
				.unwrap_or_default();
			let hf_vocab = model.get("vocab").and_then(|v| v.as_object());
			let hf_merges = model.get("merges").and_then(|v| v.as_array());

			let mut vocab: Vec<(u32, String)> = Vec::new();
			if let Some(hv) = hf_vocab {
				for (token_str, id_val) in hv {
					if let Some(id) = id_val.as_u64() {
						vocab.push((id as u32, token_str.clone()));
					}
				}
			}
			vocab.sort_by_key(|(id, _)| *id);
			let vocab: Vec<String> = vocab.into_iter().map(|(_, s)| s).collect();

			let merges: Vec<String> = hf_merges
				.map(|m| {
					m.iter()
						.filter_map(|v| v.as_str().map(|s| s.to_string()))
						.collect()
				})
				.unwrap_or_default();

			(vocab, merges)
		} else {
			(data.tokens, data.merges)
		};

		let token_type = if data.token_types.is_empty() {
			vec![0u32; vocab.len()]
		} else {
			data.token_types
		};

		let eos_token_id = data.eos_token_id;
		let pad_token_id = data.pad_token_id;

		let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(vocab.len());
		for (id, token) in vocab.iter().enumerate() {
			token_to_id.insert(token.clone(), id as u32);
		}

		for at in &data.added_tokens {
			if !token_to_id.contains_key(&at.content) {
				let id = at.id;
				if (id as usize) < vocab.len() {
					token_to_id.insert(at.content.clone(), id);
				}
			}
		}

		let mut merge_ranks: HashMap<(u32, u32), u32> = HashMap::with_capacity(merges.len());
		for (rank, merge) in merges.iter().enumerate() {
			if let Some(space_idx) = merge.find(' ') {
				let first = &merge[..space_idx];
				let second = &merge[space_idx + 1..];
				if let (Some(&id1), Some(&id2)) = (token_to_id.get(first), token_to_id.get(second))
				{
					merge_ranks.insert((id1, id2), rank as u32);
				}
			}
		}

		let (byte_to_unicode, unicode_to_byte) = Self::build_byte_mapping();

		Self {
			vocab,
			token_type,
			eos_token_id,
			pad_token_id,
			merge_ranks,
			token_to_id,
			byte_to_unicode,
			unicode_to_byte,
		}
	}

	/// Load tokenizer from a model directory.
	pub fn from_dir(dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
		let cache_path = dir.join("tokenizer.bin");
		if cache_path.exists() {
			let data = std::fs::read(&cache_path)?;
			match bincode::deserialize::<Tokenizer>(&data) {
				Ok(tok) => {
					log::info!(
						"Tokenizer: loaded from binary cache ({} tokens)",
						tok.vocab.len()
					);
					return Ok(tok);
				}
				Err(e) => {
					log::warn!(
						"Tokenizer: binary cache corrupt ({}), falling back to JSON",
						e
					);
				}
			}
		}

		let path = dir.join("tokenizer.json");
		let data = std::fs::read_to_string(&path)?;
		let raw: TokenizerData = serde_json::from_str(&data)?;
		let tok = Self::from_data(raw);

		match bincode::serialize(&tok) {
			Ok(bytes) => {
				let _ = std::fs::write(&cache_path, &bytes);
				log::info!("Tokenizer: binary cache saved ({} bytes)", bytes.len());
			}
			Err(e) => {
				log::warn!("Tokenizer: failed to save binary cache: {}", e);
			}
		}

		Ok(tok)
	}

	pub fn extract_to_file(
		gguf: &GGUFFile,
		dir: &std::path::Path,
	) -> Result<(), Box<dyn std::error::Error>> {
		let data = Self::extract_data(gguf);
		let path = dir.join("tokenizer.json");
		let json = serde_json::to_string(&data)?;
		std::fs::write(&path, json)?;
		Ok(())
	}

	fn build_byte_mapping() -> (Vec<char>, HashMap<char, u8>) {
		let mut byte_to_unicode = vec!['\0'; 256];
		let mut unicode_to_byte = HashMap::new();

		let is_self_mapped = |b: u8| -> bool {
			(33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b)
		};

		let mut n = 0u32;
		for b in 0u8..=255 {
			let codepoint = if is_self_mapped(b) {
				b as u32
			} else {
				let cp = 256 + n;
				n += 1;
				cp
			};
			let c = char::from_u32(codepoint).unwrap_or('\u{FFFD}');
			byte_to_unicode[b as usize] = c;
			unicode_to_byte.insert(c, b);
		}

		(byte_to_unicode, unicode_to_byte)
	}
	pub fn encode(&self, text: &str) -> Vec<u32> {
		if text.is_empty() {
			return Vec::new();
		}

		let pieces = Self::pretokenize(text);
		let mut all_tokens = Vec::new();

		for piece in &pieces {
			let bytes = piece.as_bytes();
			let chars: Vec<char> = bytes
				.iter()
				.map(|&b| self.byte_to_unicode[b as usize])
				.collect();

			let token_ids = self.bpe(&chars);
			all_tokens.extend(token_ids);
		}

		all_tokens
	}

	pub fn decode(&self, tokens: &[u32]) -> String {
		let mut bytes = Vec::new();
		for &tid in tokens {
			if let Some(token_str) = self.vocab.get(tid as usize) {
				for c in token_str.chars() {
					if let Some(&b) = self.unicode_to_byte.get(&c) {
						bytes.push(b);
					}
				}
			}
		}
		String::from_utf8_lossy(&bytes).into_owned()
	}

	fn pretokenize(text: &str) -> Vec<String> {
		let chars: Vec<char> = text.chars().collect();
		let n = chars.len();
		let mut tokens: Vec<String> = Vec::new();
		let mut i = 0;

		while i < n {
			let start = i;

			if chars[i] == '\'' {
				let contractions: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];
				let mut found = false;
				for suffix in &contractions {
					let suffix_chars: Vec<char> = suffix.chars().collect();
					if i + 1 + suffix_chars.len() <= n {
						let matches = suffix_chars
							.iter()
							.enumerate()
							.all(|(j, &sc)| chars[i + 1 + j] == sc);
						if matches {
							i += 1 + suffix_chars.len();
							tokens.push(chars[start..i].iter().collect());
							found = true;
							break;
						}
					}
				}
				if found {
					continue;
				}
			}

			let mut j = i;
			if j < n && chars[j] == ' ' {
				j += 1;
			}
			if j < n && chars[j].is_alphabetic() {
				while j < n && chars[j].is_alphabetic() {
					j += 1;
				}
				i = j;
				tokens.push(chars[start..i].iter().collect());
				continue;
			}
			i = start;

			j = i;
			if j < n && chars[j] == ' ' {
				j += 1;
			}
			if j < n && chars[j].is_numeric() {
				while j < n && chars[j].is_numeric() {
					j += 1;
				}
				i = j;
				tokens.push(chars[start..i].iter().collect());
				continue;
			}
			i = start;

			j = i;
			if j < n && chars[j] == ' ' {
				j += 1;
			}
			if j < n
				&& !chars[j].is_whitespace()
				&& !chars[j].is_alphabetic()
				&& !chars[j].is_numeric()
			{
				while j < n
					&& !chars[j].is_whitespace()
					&& !chars[j].is_alphabetic()
					&& !chars[j].is_numeric()
				{
					j += 1;
				}
				i = j;
				tokens.push(chars[start..i].iter().collect());
				continue;
			}
			i = start;

			if chars[i].is_whitespace() {
				let mut k = i;
				while k < n && chars[k].is_whitespace() {
					k += 1;
				}
				if k < n {
					if k > i + 1 {
						tokens.push(chars[i..k - 1].iter().collect());
					}
					i = k - 1;
				} else {
					tokens.push(chars[i..k].iter().collect());
					i = k;
				}
				continue;
			}

			i += 1;
			tokens.push(chars[start..i].iter().collect());
		}

		tokens
	}
	fn bpe(&self, chars: &[char]) -> Vec<u32> {
		let mut tokens: Vec<u32> = chars
			.iter()
			.filter_map(|c| {
				let s: String = c.to_string();
				self.token_to_id.get(&s).copied()
			})
			.collect();

		if tokens.len() < 2 {
			return tokens;
		}

		loop {
			let mut best_rank: Option<u32> = None;
			let mut best_idx: Option<usize> = None;

			for idx in 0..tokens.len().saturating_sub(1) {
				let pair = (tokens[idx], tokens[idx + 1]);
				if let Some(&rank) = self.merge_ranks.get(&pair) {
					if best_rank.map_or(true, |br| rank < br) {
						best_rank = Some(rank);
						best_idx = Some(idx);
					}
				}
			}

			let idx = match best_idx {
				Some(idx) => idx,
				None => break,
			};

			let id1 = tokens[idx];
			let id2 = tokens[idx + 1];

			let merged_str = format!(
				"{}{}",
				self.vocab
					.get(id1 as usize)
					.map(|s| s.as_str())
					.unwrap_or(""),
				self.vocab
					.get(id2 as usize)
					.map(|s| s.as_str())
					.unwrap_or(""),
			);

			if let Some(&merged_id) = self.token_to_id.get(&merged_str) {
				tokens[idx] = merged_id;
				tokens.remove(idx + 1);
			} else {
				break;
			}
		}

		tokens
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_byte_mapping_space() {
		let (b2u, u2b) = Tokenizer::build_byte_mapping();
		assert_eq!(b2u[32], '\u{120}');
		assert_eq!(u2b.get(&'\u{120}'), Some(&32u8));
	}

	#[test]
	fn test_byte_mapping_newline() {
		let (b2u, u2b) = Tokenizer::build_byte_mapping();
		assert_eq!(b2u[10], '\u{10A}');
		assert_eq!(u2b.get(&'\u{10A}'), Some(&10u8));
	}

	#[test]
	fn test_byte_mapping_printable() {
		let (b2u, _) = Tokenizer::build_byte_mapping();
		assert_eq!(b2u[33], '!');
		assert_eq!(b2u[97], 'a');
		assert_eq!(b2u[126], '~');
		assert_eq!(b2u[161], char::from_u32(161).unwrap());
		assert_eq!(b2u[172], char::from_u32(172).unwrap());
		assert_eq!(b2u[174], char::from_u32(174).unwrap());
		assert_eq!(b2u[255], char::from_u32(255).unwrap());
	}

	#[test]
	fn test_byte_mapping_control() {
		let (b2u, _) = Tokenizer::build_byte_mapping();
		assert_eq!(b2u[0], '\u{100}');
		assert_eq!(b2u[1], '\u{101}');
		assert_eq!(b2u[127], '\u{121}');
		assert_eq!(b2u[173], '\u{143}');
	}

	#[test]
	fn test_byte_mapping_completeness() {
		let (b2u, u2b) = Tokenizer::build_byte_mapping();
		for b in 0u8..=255 {
			assert_ne!(b2u[b as usize], '\0', "byte {} has null mapping", b);
			assert_eq!(u2b.get(&b2u[b as usize]), Some(&b));
		}
		assert_eq!(u2b.len(), 256);
	}

	#[test]
	fn test_pretokenize_words() {
		let pieces = Tokenizer::pretokenize("hello world");
		assert_eq!(pieces, vec!["hello", " world"]);
	}
	#[test]
	fn test_pretokenize_punctuation() {
		let pieces = Tokenizer::pretokenize("Hello, world!");
		assert_eq!(pieces, vec!["Hello", ",", " world", "!"]);
	}

	#[test]
	fn test_pretokenize_contractions() {
		let pieces = Tokenizer::pretokenize("don't");
		assert_eq!(pieces, vec!["don", "'t"]);

		let pieces = Tokenizer::pretokenize("it's a test");
		assert_eq!(pieces, vec!["it", "'s", " a", " test"]);
	}

	#[test]
	fn test_pretokenize_numbers() {
		let pieces = Tokenizer::pretokenize("abc 123");
		assert_eq!(pieces, vec!["abc", " 123"]);

		let pieces = Tokenizer::pretokenize("cost: $42.50");
		assert_eq!(pieces, vec!["cost", ":", " $", "42", ".", "50"]);
	}

	#[test]
	fn test_pretokenize_multiple_spaces() {
		let pieces = Tokenizer::pretokenize("a  b");
		assert_eq!(pieces, vec!["a", " ", " b"]);
	}

	#[test]
	fn test_pretokenize_trailing_whitespace() {
		let pieces = Tokenizer::pretokenize("abc  ");
		assert_eq!(pieces, vec!["abc", "  "]);
	}

	#[test]
	fn test_pretokenize_newline() {
		let pieces = Tokenizer::pretokenize("abc\n");
		assert_eq!(pieces, vec!["abc", "\n"]);
	}

	#[test]
	fn test_pretokenize_empty() {
		let pieces = Tokenizer::pretokenize("");
		assert!(pieces.is_empty());
	}

	#[test]
	fn test_encode_decode_roundtrip() {
		let (b2u, u2b) = Tokenizer::build_byte_mapping();

		let mut vocab: Vec<String> = Vec::new();
		let mut token_to_id: HashMap<String, u32> = HashMap::new();

		for b in 0u8..=255 {
			let c = b2u[b as usize];
			let s = c.to_string();
			let id = vocab.len() as u32;
			vocab.push(s.clone());
			token_to_id.insert(s, id);
		}

		let space_char = b2u[32];
		let space_id = token_to_id[&space_char.to_string()];
		let a_id = token_to_id[&'a'.to_string()];
		let merged = format!("{}{}", space_char, 'a');
		let merged_id = vocab.len() as u32;
		vocab.push(merged.clone());
		token_to_id.insert(merged, merged_id);

		let mut merge_ranks: HashMap<(u32, u32), u32> = HashMap::new();
		merge_ranks.insert((space_id, a_id), 0);

		let tokenizer = Tokenizer {
			vocab,
			token_type: vec![1; 257],
			eos_token_id: 0,
			pad_token_id: 0,
			merge_ranks,
			token_to_id,
			byte_to_unicode: b2u,
			unicode_to_byte: u2b,
		};

		let ids = tokenizer.encode("a");
		assert_eq!(ids, vec![a_id]);
		assert_eq!(tokenizer.decode(&ids), "a");

		let ids = tokenizer.encode(" a");
		assert_eq!(ids, vec![merged_id]);
		assert_eq!(tokenizer.decode(&ids), " a");

		let ids = tokenizer.encode("ab");
		assert_eq!(ids.len(), 2);
		assert_eq!(ids[0], a_id);
		assert_eq!(tokenizer.decode(&ids), "ab");

		let text = " a b";
		let ids = tokenizer.encode(text);
		let decoded = tokenizer.decode(&ids);
		assert_eq!(decoded, text);
	}
}

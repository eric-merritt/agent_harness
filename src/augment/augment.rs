use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Graph types ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
	pub id: Uuid,
	pub nodes: Vec<Node>,
	pub relationship: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
	pub id: Uuid,
	pub name: String,
	pub edges: Vec<Uuid>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Graph {
	pub id: Uuid,
	pub nodes: Vec<Node>,
	pub edges: Vec<Edge>,
}

impl Graph {
	pub fn new(id: Uuid, nodes: Vec<Node>, edges: Vec<Edge>) -> Self {
		Self { id, nodes, edges }
	}
}

// ── Prompt building blocks ─────────────────────────────────────────────────────

/// A conditional rule attached to a prompt block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
	pub id: Uuid,
	pub content: String,
	pub is_active: bool,
}

/// A personality trait injected into the system prompt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersonalityTrait {
	pub id: Uuid,
	pub content: String,
	pub is_active: bool,
}

/// A few-shot example for in-context learning.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Example {
	pub id: Uuid,
	pub input: String,
	pub output: String,
	pub is_active: bool,
}

/// A context snippet — codebase tidbit, doc excerpt, or factual note.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextSnippet {
	pub id: Uuid,
	pub content: String,
	pub source: String,
	pub token_estimate: usize,
	pub is_active: bool,
}

/// An atomic piece of a prompt.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PromptBlock {
	Rule(Rule),
	PersonalityTrait(PersonalityTrait),
	Example(Example),
	ContextSnippet(ContextSnippet),
}

/// A composed system prompt built from active prompt blocks.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SystemPrompt {
	pub blocks: Vec<PromptBlock>,
}

impl SystemPrompt {
	/// Render the active blocks into a single string for the API call.
	pub fn render(&self) -> String {
		let mut out = String::new();
		for block in &self.blocks {
			let active = match block {
				PromptBlock::Rule(r) => r.is_active,
				PromptBlock::PersonalityTrait(p) => p.is_active,
				PromptBlock::Example(e) => e.is_active,
				PromptBlock::ContextSnippet(c) => c.is_active,
			};
			if !active {
				continue;
			}
			match block {
				PromptBlock::Rule(r) => {
					out.push_str(&format!("Rule: {}\n", r.content));
				}
				PromptBlock::PersonalityTrait(p) => {
					out.push_str(&format!("Trait: {}\n", p.content));
				}
				PromptBlock::Example(e) => {
					out.push_str(&format!("Input: {}\nOutput: {}\n", e.input, e.output));
				}
				PromptBlock::ContextSnippet(c) => {
					out.push_str(&format!("Context ({}): {}\n", c.source, c.content));
				}
			}
		}
		out
	}
}

// ── Attachment ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum AttachmentKind {
	Image {
		mime_type: String,
		width: Option<u32>,
		height: Option<u32>,
	},
	Code {
		language: String,
		mime_type: Option<String>,
	},
	Document {
		mime_type: String,
		page_count: Option<u32>,
	},
	Audio {
		mime_type: String,
		duration_secs: Option<f64>,
	},
	Video {
		mime_type: String,
		width: Option<u32>,
		height: Option<u32>,
		duration_secs: Option<f64>,
	},
	Blob {
		mime_type: Option<String>,
	},
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attachment {
	pub id: Uuid,
	pub kind: AttachmentKind,
	pub content: String,
	pub size_bytes: u64,
}

// ── Task ───────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
	Pending,
	InProgress,
	UpNext,
	Complete,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Task {
	pub id: Uuid,
	pub description: String,
	pub status: TaskStatus,
	pub depends_on: Vec<Uuid>,
	pub priority: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskList {
	pub id: Uuid,
	pub tasks: Vec<Task>,
}

impl TaskList {
	pub fn new(tasks: Vec<Task>) -> Self {
		Self {
			id: Uuid::new_v4(),
			tasks,
		}
	}

	pub fn ready_tasks(&self) -> Vec<&Task> {
		let completed: std::collections::HashSet<Uuid> = self
			.tasks
			.iter()
			.filter(|t| t.status == TaskStatus::Complete)
			.map(|t| t.id)
			.collect();

		self.tasks
			.iter()
			.filter(|t| t.status == TaskStatus::Pending || t.status == TaskStatus::UpNext)
			.filter(|t| t.depends_on.iter().all(|dep| completed.contains(dep)))
			.collect()
	}
}

// ── LoRA ───────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoRa {
	pub id: Uuid,
	pub base_model: String,
	pub adapter_path: String,
	pub rank: u32,
	pub target_modules: Vec<String>,
	pub alpha: f32,
	pub dropout: f32,
}

// ── MCP Server augment ────────────────────────────────────────────────────────

/// A connected MCP server augmenting the model with remote tools.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpServerAugment {
	pub id: Uuid,
	pub name: String,
	pub endpoint: String,
	pub transport: String,
	pub connected: bool,
	pub available_tools: Vec<String>,
}

/// A tool group from the MCP server, pluggable as a unit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolGroupAugment {
	pub id: Uuid,
	pub group_name: String,
	pub tools: Vec<String>,
	pub source_server: Uuid,
}

// ── Loop augment ───────────────────────────────────────────────────────────────

/// Trigger condition for a loop augment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LoopTrigger {
	/// Fire every N seconds
	Interval { secs: u64 },
	/// Fire when a condition prompt evaluates true
	Conditional { prompt: String },
	/// Fire on a specific event
	EventDriven { event: String },
	/// Self-paced — the loop decides when to re-fire
	SelfPaced,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopAugment {
	pub id: Uuid,
	pub name: String,
	pub prompt: String,
	pub trigger: LoopTrigger,
	pub max_iterations: Option<u32>,
	pub is_active: bool,
}

// ── Tensor / Weight management ─────────────────────────────────────────────────

/// Data type of a tensor as stored in the safetensors file.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TensorDtype {
	F32,
	F16,
	BF16,
	I8,
	I16,
	I32,
	I64,
	U8,
	Bool,
}

/// Where a tensor currently lives in the memory hierarchy.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TensorPlacement {
	/// Device-local VRAM — fastest for compute
	Gpu,
	/// System RAM — host-visible, slower compute
	Cpu,
	/// Mmap'd from safetensors file — zero-copy on first access, paged by OS
	Disk,
	/// Pinned host memory — for async transfers to GPU
	Staging,
}

/// Metadata describing a single tensor in a model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorDescriptor {
	pub name: String,
	pub shape: Vec<usize>,
	pub dtype: TensorDtype,
	pub byte_offset: u64,
	pub byte_size: u64,
	/// Which layer this tensor belongs to (parsed from name).
	pub layer_index: Option<usize>,
}

impl TensorDescriptor {
	/// Element count = product of shape dims.
	pub fn num_elements(&self) -> usize {
		self.shape.iter().product()
	}

	/// Bytes per element for this dtype.
	pub fn bytes_per_element(&self) -> usize {
		match self.dtype {
			TensorDtype::F32 | TensorDtype::I32 => 4,
			TensorDtype::F16 | TensorDtype::BF16 | TensorDtype::I16 => 2,
			TensorDtype::I8 | TensorDtype::U8 | TensorDtype::Bool => 1,
			TensorDtype::I64 => 8,
		}
	}

	/// Verify that shape * dtype matches byte_size.
	pub fn is_consistent(&self) -> bool {
		self.num_elements() * self.bytes_per_element() == self.byte_size as usize
	}

	/// Parse layer index from tensor name (e.g., "model.layers.5.self_attn.q_proj.weight" → 5).
	pub fn parse_layer_index(name: &str) -> Option<usize> {
		let parts: Vec<&str> = name.split('.').collect();
		for (i, part) in parts.iter().enumerate() {
			if *part == "layers" {
				return parts.get(i + 1).and_then(|s| s.parse().ok());
			}
		}
		None
	}
}

/// A parsed safetensors header — the full tensor map for a model.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelTensorMap {
	pub model_name: String,
	pub tensors: Vec<TensorDescriptor>,
	pub total_bytes: u64,
}

impl ModelTensorMap {
	/// Total VRAM needed if all tensors go to GPU.
	pub fn total_vram_needed(&self) -> u64 {
		self.total_bytes
	}

	/// Tensors for a specific layer.
	pub fn layer_tensors(&self, layer: usize) -> Vec<&TensorDescriptor> {
		self.tensors
			.iter()
			.filter(|t| t.layer_index == Some(layer))
			.collect()
	}

	/// Embedding tensors (lm_head, embed_tokens).
	pub fn embedding_tensors(&self) -> Vec<&TensorDescriptor> {
		self.tensors
			.iter()
			.filter(|t| t.name.contains("embed") || t.name.contains("lm_head"))
			.collect()
	}

	/// Count of layers in the model.
	pub fn layer_count(&self) -> usize {
		self.tensors
			.iter()
			.filter_map(|t| t.layer_index)
			.max()
			.map(|m| m + 1)
			.unwrap_or(0)
	}
}

/// Quantization precision applied to a tensor during compression.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum QuantPrecision {
	/// Full precision — no quantization
	Fp32,
	/// Half precision (IEEE 754)
	Fp16,
	/// Brain float 16
	Bf16,
	/// 8-bit integer quantization
	Int8,
	/// 4-bit block-wise quantization
	Int4,
}

impl QuantPrecision {
	/// Estimated bytes per element after quantization.
	pub fn bytes_per_element(&self) -> f32 {
		match self {
			QuantPrecision::Fp32 => 4.0,
			QuantPrecision::Fp16 | QuantPrecision::Bf16 => 2.0,
			QuantPrecision::Int8 => 1.0,
			QuantPrecision::Int4 => 0.5,
		}
	}

	/// Compressed size estimate for a tensor of the given byte count.
	/// Input is the original (typically F32) byte count.
	pub fn compressed_size(&self, original_bytes: u64) -> u64 {
		let ratio = self.bytes_per_element() / 4.0; // assume original is F32
		(original_bytes as f32 * ratio) as u64
	}
}

/// Placement decision for a single tensor — where it should live.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorPlacementRule {
	pub tensor_name: String,
	pub placement: TensorPlacement,
	pub pinned: bool,
	/// Quantization precision to apply when compressing this tensor.
	pub quant_precision: QuantPrecision,
}

/// A complete weight placement plan for a model.
/// Decides which tensors go to GPU, CPU, or Disk based on available memory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightPlacementPlan {
	pub model: String,
	pub rules: Vec<TensorPlacementRule>,
	pub gpu_budget_bytes: u64,
	pub cpu_budget_bytes: u64,
	pub kv_cache_placement: TensorPlacement,
}

impl WeightPlacementPlan {
	/// Default quantization for regular layer weights.
	const DEFAULT_QUANT: QuantPrecision = QuantPrecision::Int4;
	/// Higher-precision quantization for the LM head — still compressed, but
	/// kept above the layer-weight precision to preserve the output distribution.
	const LM_HEAD_QUANT: QuantPrecision = QuantPrecision::Int8;
	/// Norm tensors and other small weights kept at half precision for stability.
	const NORM_QUANT: QuantPrecision = QuantPrecision::Fp16;

	/// Plan placement: hot layers on GPU, cold layers on CPU, overflow on disk.
	/// Quantization is applied per-tensor: LM head at Int8, layer weights at Int4,
	/// norm tensors at Fp16. GPU budget is tracked against compressed sizes.
	pub fn plan(tensor_map: &ModelTensorMap, gpu_available: u64, cpu_available: u64) -> Self {
		let gpu_budget = (gpu_available as f64 * 0.8) as u64; // 20% headroom
		let mut rules = Vec::new();
		let mut gpu_used: u64 = 0;

		let n_layers = tensor_map.layer_count();

		// Phase 1: Place embeddings + LM head on GPU (always needed).
		// LM head gets Int8 (higher precision), embeddings get Int4.
		for t in &tensor_map.tensors {
			if t.name.contains("embed") || t.name.contains("lm_head") {
				let quant = if t.name.contains("lm_head") {
					Self::LM_HEAD_QUANT
				} else {
					Self::DEFAULT_QUANT
				};
				let compressed = quant.compressed_size(t.byte_size);
				if gpu_used + compressed <= gpu_budget {
					rules.push(TensorPlacementRule {
						tensor_name: t.name.clone(),
						placement: TensorPlacement::Gpu,
						pinned: true,
						quant_precision: quant,
					});
					gpu_used += compressed;
				} else {
					rules.push(TensorPlacementRule {
						tensor_name: t.name.clone(),
						placement: TensorPlacement::Cpu,
						pinned: true,
						quant_precision: quant,
					});
				}
			}
		}

		// Phase 2: Place layer weights — front layers on GPU, rest on CPU.
		// All layer weights quantized to Int4.
		for t in &tensor_map.tensors {
			if t.layer_index.is_none() {
				continue;
			}
			let layer = t.layer_index.unwrap();
			let is_hot = layer < n_layers / 2;
			let compressed = Self::DEFAULT_QUANT.compressed_size(t.byte_size);

			if is_hot && gpu_used + compressed <= gpu_budget {
				rules.push(TensorPlacementRule {
					tensor_name: t.name.clone(),
					placement: TensorPlacement::Gpu,
					pinned: false,
					quant_precision: Self::DEFAULT_QUANT,
				});
				gpu_used += compressed;
			} else {
				rules.push(TensorPlacementRule {
					tensor_name: t.name.clone(),
					placement: TensorPlacement::Cpu,
					pinned: false,
					quant_precision: Self::DEFAULT_QUANT,
				});
			}
		}

		// Phase 3: Place remaining tensors (norm, etc.) on GPU if space.
		// Norm tensors kept at Fp16 for stability.
		for t in &tensor_map.tensors {
			if rules.iter().any(|r| r.tensor_name == t.name) {
				continue;
			}
			let compressed = Self::NORM_QUANT.compressed_size(t.byte_size);
			if gpu_used + compressed <= gpu_budget {
				rules.push(TensorPlacementRule {
					tensor_name: t.name.clone(),
					placement: TensorPlacement::Gpu,
					pinned: true,
					quant_precision: Self::NORM_QUANT,
				});
				gpu_used += compressed;
			} else {
				rules.push(TensorPlacementRule {
					tensor_name: t.name.clone(),
					placement: TensorPlacement::Cpu,
					pinned: true,
					quant_precision: Self::NORM_QUANT,
				});
			}
		}

		// KV-cache placement: GPU if room, else CPU, else disk
		let kv_placement = if gpu_used + (n_layers as u64 * 1024 * 1024) <= gpu_budget {
			TensorPlacement::Gpu
		} else {
			TensorPlacement::Cpu
		};

		Self {
			model: tensor_map.model_name.clone(),
			rules,
			gpu_budget_bytes: gpu_budget,
			cpu_budget_bytes: cpu_available,
			kv_cache_placement: kv_placement,
		}
	}
}

// ── Weight editing (abliteration) ──────────────────────────────────────────────

/// Identifies which weights to modify and how.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AblationMethod {
	/// Zero out a direction in weight space (refusal direction removal)
	DirectionRemoval { direction: Vec<f32>, scale: f32 },
	/// SVD-based steering: project out a subspace
	SvdSteering { u: Vec<Vec<f32>>, scale: f32 },
	/// Direct weight replacement
	DirectReplace { new_values: Vec<f32> },
	/// Additive delta
	AddDelta { delta: Vec<f32> },
}

/// A single weight edit operation for an abliteration pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightEdit {
	pub id: Uuid,
	pub tensor_name: String,
	pub method: AblationMethod,
	pub layer_index: Option<usize>,
	pub description: String,
	pub applied: bool,
}

/// A collection of weight edits forming an abliteration pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AblationPipeline {
	pub id: Uuid,
	pub name: String,
	pub edits: Vec<WeightEdit>,
}

impl AblationPipeline {
	/// Apply edits in order to a weight buffer.
	/// Returns the modified buffer.
	pub fn apply(&self, tensor_name: &str, weights: &mut [f32]) {
		for edit in &self.edits {
			if edit.applied {
				continue;
			}
			if edit.tensor_name != tensor_name {
				continue;
			}
			match &edit.method {
				AblationMethod::DirectionRemoval { direction, scale } => {
					// Project out the direction: w = w - scale * (w . d) * d
					let dot: f32 = weights
						.iter()
						.zip(direction.iter())
						.map(|(w, d)| w * d)
						.sum();
					for (w, d) in weights.iter_mut().zip(direction.iter()) {
						*w -= scale * dot * d;
					}
				}
				AblationMethod::AddDelta { delta } => {
					for (w, d) in weights.iter_mut().zip(delta.iter()) {
						*w += d;
					}
				}
				AblationMethod::DirectReplace { new_values } => {
					for (w, v) in weights.iter_mut().zip(new_values.iter()) {
						*w = *v;
					}
				}
				AblationMethod::SvdSteering { u, scale } => {
					// w = w - scale * U @ U^T @ w (project out subspace)
					// Simplified: for each row u_i, subtract projection
					for u_row in u {
						let proj: f32 =
							weights.iter().zip(u_row.iter()).map(|(w, ui)| w * ui).sum();
						for (w, ui) in weights.iter_mut().zip(u_row.iter()) {
							*w -= scale * proj * ui;
						}
					}
				}
			}
		}
	}
}

// ── Compressed model weights ──────────────────────────────────────────────────

/// Compressed model weights in DedupCountTensor format.
/// Plugged into the AugmentBus to provide on-demand weight decompression.
///
/// The actual weight data lives in two sidecar files:
///   - weights.bin  — core compressed data (prefixes, unique tails, counts)
///   - sandbag.bin  — per-weight metadata (group assignments, sign bits)
///
/// A manifest.json provides per-tensor offsets and sizes for random access.
/// The WeightPlacementPlan decides where each tensor lives (GPU/CPU/Disk);
/// this augment provides HOW to get the data (decompress from files).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedModelAugment {
	pub id: Uuid,
	pub model_name: String,
	pub weights_path: String,
	pub sandbag_path: String,
	pub manifest_path: String,
	pub tensor_count: usize,
	pub total_core_bytes: u64,
	pub total_sandbag_bytes: u64,
	/// Original model size in bytes (as f32).
	pub total_original_bytes: u64,
	/// Core compression ratio (original / core).
	pub core_ratio: f32,
	pub is_active: bool,
}

// ── Top-level augment enum ─────────────────────────────────────────────────────

/// Every way we can augment an LLM's capabilities or context.
/// Each variant is an atomic, pluggable unit that can be hot-swapped at any time.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "augment_type", rename_all = "camelCase")]
pub enum Augment {
	Tool { name: String },
	PromptBlock(PromptBlock),
	SystemPrompt(SystemPrompt),
	Attachment(Attachment),
	TaskList(TaskList),
	LoRa(LoRa),
	McpServer(McpServerAugment),
	ToolGroup(ToolGroupAugment),
	Loop(LoopAugment),
	TensorMap(ModelTensorMap),
	WeightPlan(WeightPlacementPlan),
	AblationPipeline(AblationPipeline),
	WeightEdit(WeightEdit),
	CompressedModel(CompressedModelAugment),
}

/// A hot-pluggable augment slot on the app surface.
pub struct AugmentSlot {
	pub id: Uuid,
	pub augment: Augment,
	pub is_plugged: bool,
	pub priority: u8,
}

impl AugmentSlot {
	pub fn plug(&mut self) {
		self.is_plugged = true;
		log::info!("Augment slot {} plugged", self.id);
	}
	pub fn unplug(&mut self) {
		self.is_plugged = false;
		log::info!("Augment slot {} unplugged", self.id);
	}
}

/// The augment bus — manages all plugged augments.
pub struct AugmentBus {
	pub slots: Vec<AugmentSlot>,
}

impl AugmentBus {
	pub fn new() -> Self {
		Self { slots: Vec::new() }
	}

	pub fn plug(&mut self, augment: Augment, priority: u8) -> Uuid {
		let id = Uuid::new_v4();
		log::info!(
			"AugmentBus: plugging augment id={} priority={}",
			id,
			priority
		);
		self.slots.push(AugmentSlot {
			id,
			augment,
			is_plugged: true,
			priority,
		});
		self.slots.sort_by_key(|s| s.priority);
		log::debug!(
			"AugmentBus: slots re-sorted by priority — {} total slots",
			self.slots.len()
		);
		id
	}

	pub fn unplug(&mut self, id: Uuid) {
		if let Some(slot) = self.slots.iter_mut().find(|s| s.id == id) {
			slot.unplug();
		} else {
			log::warn!("AugmentBus: unplug called for unknown slot id={}", id);
		}
	}

	pub fn active_prompt_blocks(&self) -> Vec<&PromptBlock> {
		self.slots
			.iter()
			.filter(|s| s.is_plugged)
			.filter_map(|s| match &s.augment {
				Augment::PromptBlock(pb) => Some(pb),
				_ => None,
			})
			.collect()
	}

	pub fn active_tools(&self) -> Vec<&str> {
		self.slots
			.iter()
			.filter(|s| s.is_plugged)
			.filter_map(|s| match &s.augment {
				Augment::Tool { name } => Some(name.as_str()),
				_ => None,
			})
			.collect()
	}

	pub fn active_mcp_servers(&self) -> Vec<&McpServerAugment> {
		self.slots
			.iter()
			.filter(|s| s.is_plugged)
			.filter_map(|s| match &s.augment {
				Augment::McpServer(mcp) => Some(mcp),
				_ => None,
			})
			.collect()
	}

	pub fn active_loops(&self) -> Vec<&LoopAugment> {
		self.slots
			.iter()
			.filter(|s| s.is_plugged)
			.filter_map(|s| match &s.augment {
				Augment::Loop(l) => Some(l),
				_ => None,
			})
			.collect()
	}

	pub fn weight_plan(&self) -> Option<&WeightPlacementPlan> {
		self.slots
			.iter()
			.filter(|s| s.is_plugged)
			.find_map(|s| match &s.augment {
				Augment::WeightPlan(p) => Some(p),
				_ => None,
			})
	}

	pub fn ablation_pipelines(&self) -> Vec<&AblationPipeline> {
		self.slots
			.iter()
			.filter(|s| s.is_plugged)
			.filter_map(|s| match &s.augment {
				Augment::AblationPipeline(p) => Some(p),
				_ => None,
			})
			.collect()
	}

	pub fn tensor_map(&self) -> Option<&ModelTensorMap> {
		self.slots
			.iter()
			.filter(|s| s.is_plugged)
			.find_map(|s| match &s.augment {
				Augment::TensorMap(m) => Some(m),
				_ => None,
			})
	}

	pub fn compressed_model(&self) -> Option<&CompressedModelAugment> {
		self.slots
			.iter()
			.filter(|s| s.is_plugged)
			.find_map(|s| match &s.augment {
				Augment::CompressedModel(m) => Some(m),
				_ => None,
			})
	}
}

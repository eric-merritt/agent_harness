//! Tensor Communications Server — the global state object for a loaded model.
//!
//! This is the single owner of everything that exists once per model session:
//! the arena layout, every pipeline, every descriptor set, and the residency
//! policy. Nothing in this module is created at dispatch time; everything is
//! built here, at load time, and reused for the life of the session.
//!
//! The three binding categories (weights / augments / shaders) are all
//! resolved against the same virtual tensor arena, so a shader launch is just:
//! bind pipeline → push constants → dispatch. No file I/O, no descriptor
//! allocation, no residency decisions at call time.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::inference::config::ModelConfig;
use crate::memory_controller::controller::GpuContext;
use crate::memory_controller::virtual_tensor_arena::{ArenaRegion, RegionCategory, VirtualTensorArena};

// ──────────────────────────────────────────────────────────────────────────────
// 1. ARENA LAYOUT
// ──────────────────────────────────────────────────────────────────────────────
// ArenaRegion, RegionCategory, and SourceSpan live in virtual_tensor_arena.rs —
// they are the arena's own vocabulary for describing its address space.

// ──────────────────────────────────────────────────────────────────────────────
// 2. PIPELINE REGISTRY — every shader, compiled once at load time
// ──────────────────────────────────────────────────────────────────────────────

/// A fully-built compute pipeline ready for dispatch.
///
/// This is the direct generalization of `HessianPipeline` in hessian.rs:
/// same fields, but owned by the server rather than created ad-hoc at call
/// time. The descriptor set here is pre-built and immutable — it points at
/// virtual arena addresses that are stable for the session's lifetime.
pub struct PipelineEntry {
	/// The compiled compute pipeline (vk::Pipeline).
	pub pipeline: ash::vk::Pipeline,
	/// Its layout (push constant range + descriptor set layout).
	pub layout: ash::vk::PipelineLayout,
	/// Descriptor set layout — kept separately so we can allocate replacement
	/// sets if a region's backing buffer changes without rebuilding the pipeline.
	pub set_layout: ash::vk::DescriptorSetLayout,
	/// The pre-built descriptor set. Immutable after load unless a region is
	/// re-bound (see `rebind_region`).
	pub set: ash::vk::DescriptorSet,
}

/// All pipelines for the loaded model, keyed by logical operation name.
///
/// Built once in `ModelServer::build_pipelines()`. The key space is fixed by
/// the model architecture — e.g. "gemv", "rms_norm", "rope", "attn_fwd",
/// "swiglu_ffn" — not by tensor, so one pipeline serves every layer of its kind.
pub struct PipelineRegistry {
	/// Operation name → built pipeline.
	pub entries: HashMap<String, PipelineEntry>,
	/// Shared descriptor pool all sets are allocated from. Freed once at shutdown.
	pub pool: ash::vk::DescriptorPool,
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. RESIDENCY POLICY — who decides which pages are on the GPU
// ──────────────────────────────────────────────────────────────────────────────

/// The residency state of every region in the arena at any point in time.
///
/// This is NOT the same as `VirtualTensorArena::page_table` (which tracks
/// individual pages). This is the policy layer: it knows which *regions* are
/// expected to be resident, and drives commit/evict calls against the arena
/// when that expectation changes.
pub struct ResidencyTable {
	/// Region name → current residency state as tracked by the policy.
	pub states: HashMap<String, RegionResidency>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegionResidency {
	/// No pages mapped. First access triggers a full commit.
	Cold,
	/// All (or most) pages are GpuResident. Ready for dispatch.
	Resident,
	/// Pages have been evicted to CPU backing. Must be re-committed before use.
	Evicted,
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. THE SERVER — the global state object
// ──────────────────────────────────────────────────────────────────────────────

/// The tensor communications server.
///
/// One instance per loaded model. Owns the arena, the pipeline registry, and
/// the residency policy. All public methods are idempotent with respect to
/// load: calling `load_model` twice on the same server is an error, not a
/// silent re-load.
pub struct ModelServer {
	/// The model's architectural parameters (layer count, head dims, etc.).
	pub config: ModelConfig,

	/// The virtual tensor arena — the single sparse buffer all data lives in.
	pub arena: VirtualTensorArena,

	/// The planned layout: which region owns which virtual address range.
	/// Built once in `plan_layout()`, never mutated after.
	pub layout: Vec<ArenaRegion>,

	/// All compiled pipelines, keyed by operation name.
	pub pipelines: PipelineRegistry,

	/// Current residency state per region.
	pub residency: ResidencyTable,

	/// The GPU context (device, queue, allocator). Shared with the rest of the
	/// app; the server does not own it, only borrows it for dispatch.
	pub gpu: GpuContext,

	/// Path to the backing model file (sandbag / safetensors). Kept so lazy
	/// loads can seek into it without re-opening.
	source_path: PathBuf,
}

impl ModelServer {
	// ── CONSTRUCTION ──────────────────────────────────────────────────────────

	/// Plan the arena layout for a given model config WITHOUT creating any GPU
	/// resources yet. Returns the region list that `new()` will use to size
	/// and carve up the sparse buffer.
	///
	/// What this must accomplish:
	/// - Walk every tensor name in the model (from the sandbag index or a
	///   hardcoded per-architecture template) and assign each a virtual offset.
	/// - Round each region's size up to a whole number of arena pages.
	/// - Reserve activation regions (KV cache, scratch planes) at fixed offsets
	///   so their addresses are stable across forward passes.
	/// - Return regions in commit-priority order: weights first, augments
	///   second, activations last — this is the eviction order under OOM.
	pub fn plan_layout(config: &ModelConfig) -> Vec<ArenaRegion> {
		todo!("Walk model tensors, assign virtual offsets, return region list")
	}

	/// Create the server and build all GPU resources for the given model.
	///
	/// This is the expensive call — it runs once per model load. It must:
	/// 1. Call `plan_layout()` to get the region list.
	/// 2. Create the `VirtualTensorArena` sized to fit all regions.
	/// 3. Build every pipeline in `build_pipelines()`.
	/// 4. Pre-build every descriptor set, binding each to its region's virtual
	///    address (the sparse buffer handle + offset — no data copy needed).
	/// 5. Run `validate_layout()` and abort on any inconsistency.
	///
	/// After this returns, the server is ready for dispatch with zero further
	/// setup cost per forward pass.
	pub fn new(
		config: &ModelConfig,
		source_path: PathBuf,
		gpu: GpuContext,
	) -> Result<Self, ModelServerError> {
		todo!("Create arena, build pipelines, pre-build descriptor sets, validate")
	}

	// ── LAYOUT VALIDATION ─────────────────────────────────────────────────────

	/// Check the planned layout for internal consistency before any GPU work.
	///
	/// What this must catch:
	/// - Two regions overlapping in virtual address space.
	/// - A region whose byte_size is not a multiple of page_size.
	/// - Total arena size exceeding the sparse buffer's max size query.
	/// - A region with category `Weights` that has no `SourceSpan` (can't lazy-load).
	///
	/// Returns `Err` with a human-readable description of the first violation.
	pub fn validate_layout(layout: &[ArenaRegion], page_size: u64) -> Result<(), ModelServerError> {
		todo!("Check for overlaps, alignment, size limits")
	}

	// ── PIPELINE BUILDING ─────────────────────────────────────────────────────

	/// Build every compute pipeline the model needs and store them in the registry.
	///
	/// What this must accomplish:
	/// - For each operation type the architecture requires (GEMV, RMSNorm,
	///   RoPE, attention forward, SwiGLU FFN, etc.), read the .spv file from
	///   disk ONCE, create the shader module, pipeline layout, and pipeline.
	/// - Allocate one descriptor set per pipeline from the shared pool, binding:
	///     binding 0 → activation/scratch arena region (virtual address)
	///     binding 1 → weight arena region (virtual address)
	///   The exact binding count depends on the shader; check each .comp file.
	/// - Store everything in `self.pipelines.entries`.
	///
	/// After this call, no pipeline is ever created again for this model.
	pub fn build_pipelines(&mut self) -> Result<(), ModelServerError> {
		todo!("Read .spv files, create pipelines, allocate and bind descriptor sets")
	}

	// ── RESIDENCY MANAGEMENT ──────────────────────────────────────────────────

	/// Ensure a region's pages are resident on the GPU before dispatch.
	///
	/// This is the ONLY place residency decisions are made at runtime. The
	/// forward pass calls this before each layer; if the region is already
	/// `Resident` it is a no-op. If `Cold` or `Evicted`, it commits pages via
	/// `arena.commit_page()` until the region is fully mapped.
	///
	/// What this must accomplish:
	/// - Check `self.residency.states[name]`. Return immediately if `Resident`.
	/// - For each page in the region's virtual range, call `arena.commit_page()`.
	///   The arena handles OOM → CPU fallback internally.
	/// - Update `self.residency.states[name]` to `Resident` on success.
	/// - Return `Err` if a page could not be committed and no CPU fallback
	///   was possible (true OOM with no host memory).
	pub fn ensure_resident(&mut self, region_name: &str) -> Result<(), ModelServerError> {
		todo!("Commit pages for the named region; no-op if already resident")
	}

	/// Evict a region's pages from GPU memory, freeing VRAM for other regions.
	///
	/// Called by the OOM handler or by an explicit memory-pressure signal.
	/// Pages are unbound (not freed) — their CPU backing remains valid so a
	/// future `ensure_resident` can re-commit without re-reading from disk.
	///
	/// What this must accomplish:
	/// - For each page in the region's virtual range, call `arena.evict_page()`.
	/// - Update `self.residency.states[name]` to `Evicted`.
	/// - Never evict a region with category `Activation` (it will be overwritten
	///   on the next forward pass anyway; evicting it is wasted work).
	pub fn evict_region(&mut self, region_name: &str) -> Result<(), ModelServerError> {
		todo!("Unbind pages for the named region; mark Evicted")
	}

	/// Re-bind a descriptor set's buffer binding to point at a different virtual
	/// offset. Used when a region is moved (e.g. after a layout migration).
	///
	/// What this must accomplish:
	/// - Find the pipeline entry whose descriptor set binds `region_name`.
	/// - Call `vk::update_descriptor_sets` with the new offset, same buffer handle.
	/// - This is rare; in steady state bindings are stable for the session.
	pub fn rebind_region(&mut self, region_name: &str, new_virtual_offset: u64) -> Result<(), ModelServerError> {
		todo!("vk::update_descriptor_sets with new offset")
	}

	// ── DISPATCH ──────────────────────────────────────────────────────────────

	/// Launch a single pipeline against the arena.
	///
	/// This is the hot path — called once per operation per forward pass.
	/// It must be as cheap as possible: no allocation, no file I/O, no
	/// residency decisions (those happen in `ensure_resident` before this call).
	///
	/// What this must accomplish:
	/// - Look up the pipeline entry by name (HashMap get — O(1)).
	/// - Record into a command buffer: bind pipeline, push constants, dispatch.
	/// - The descriptor set is already bound in the pipeline entry; no update needed.
	/// - Return the command buffer handle for the caller to submit.
	///
	/// `push_constants` is operation-specific (tile offsets, element counts,
	/// etc.) and varies per call even for the same pipeline.
	pub fn dispatch(
		&self,
		pipeline_name: &str,
		push_constants: &[u32],
		dispatch_size: [u32; 3],
	) -> Result<ash::vk::CommandBuffer, ModelServerError> {
		todo!("Bind pipeline, push constants, dispatch — no allocation")
	}

	// ── SHUTDOWN ──────────────────────────────────────────────────────────────

	/// Tear down all GPU resources in the correct order.
	///
	/// What this must accomplish:
	/// - Destroy all descriptor sets (via the pool).
	/// - Destroy all pipelines and their layouts.
	/// - Destroy the sparse buffer (arena).
	/// - Order matters: descriptor sets before pools, pipelines before layouts,
	///   buffers last. Vulkan requires this destruction order.
	pub fn shutdown(&mut self) {
		todo!("Destroy in reverse creation order")
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. ERROR TYPE
// ──────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during server construction or operation.
#[derive(Debug)]
pub enum ModelServerError {
	/// Layout validation failed (overlap, misalignment, size overflow).
	Layout(String),
	/// A pipeline failed to compile from its .spv file.
	PipelineBuild { name: String, cause: String },
	/// A page could not be committed and no CPU fallback was available.
	Oom { region: String, page_index: u64 },
	/// The requested operation has no registered pipeline.
	UnknownPipeline(String),
	/// The requested region does not exist in the layout.
	UnknownRegion(String),
	/// A Vulkan call failed.
	Vk(ash::vk::Result),
}

impl std::fmt::Display for ModelServerError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Layout(msg) => write!(f, "layout error: {msg}"),
			Self::PipelineBuild { name, cause } => write!(f, "pipeline '{name}' failed to build: {cause}"),
			Self::Oom { region, page_index } => write!(f, "OOM committing region '{region}' page {page_index}"),
			Self::UnknownPipeline(name) => write!(f, "no pipeline registered for '{name}'"),
			Self::UnknownRegion(name) => write!(f, "no region named '{name}' in layout"),
			Self::Vk(r) => write!(f, "Vulkan error: {r:?}"),
		}
	}
}

impl std::error::Error for ModelServerError {}

// ──────────────────────────────────────────────────────────────────────────────
// 6. FORWARD-PASS ENTRY POINT (skeleton only — lives here or in mod.rs)
// ──────────────────────────────────────────────────────────────────────────────

impl ModelServer {
	/// Run one full forward pass for a single token.
	///
	/// This is the function `InferenceEngine::forward()` will eventually call
	/// instead of doing its own buffer management. It must:
	/// 1. For each layer, in order:
	///    a. `ensure_resident("blk.{i}.attn_q.weight")` (and other weight regions)
	///    b. `dispatch("rms_norm", ...)` for the input norm
	///    c. `dispatch("gemv", ...)` for QKV projection
	///    d. `dispatch("rope", ...)` on Q and K
	///    e. `dispatch("attn_fwd", ...)` — reads/writes KV arena regions
	///    f. `dispatch("gemv", ...)` for W_O
	///    g. residual add (can be fused into the GEMV epilogue or a separate kernel)
	///    h. `dispatch("rms_norm", ...)` for the FFN input norm
	///    i. `dispatch("swiglu_ffn", ...)` — gate + up + activation in one pass
	///    j. residual add
	/// 2. After all layers: `dispatch("rms_norm", ...)` final norm, then
	///    `dispatch("gemv", ...)` for lm_head → logits.
	/// 3. Return the logits vector (or a handle to it in the arena).
	///
	/// The KV cache regions are pre-planned in the layout; their addresses are
	/// stable so the attention pipeline's descriptor set never needs updating.
	pub fn forward(&mut self, token_id: u32) -> Result<Vec<f32>, ModelServerError> {
		todo!("Per-layer dispatch loop; return logits")
	}
}

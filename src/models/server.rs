// Model server — connects the augment bus, model loader, weight placement,
// and virtual device into a unified runtime.
//
// The server opens a compressed model directory, plugs augments into the bus,
// and provides tensor access through the placement plan.
//
// Architecture:
//   AugmentBus
//     ├── CompressedModelAugment (what model, where files are)
//     ├── WeightPlacementPlan (GPU/CPU/Disk per tensor)
//     └── TensorMap (shape/dtype metadata)
//   ModelLoader (reads weights.bin + sandbag.bin + manifest.json)
//   VirtualDevice (unified GPU VRAM + CPU RAM address space)

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use super::convert::common::{ConversionStats, TensorStats};
use super::convert::loader::ModelLoader;
use crate::augment::augment::{
	Augment, AugmentBus, CompressedModelAugment, ModelTensorMap, TensorDescriptor, TensorDtype,
	TensorPlacement, WeightPlacementPlan,
};
use crate::inference::InferenceEngine;

/// Running model server. Holds the augment bus, loader, and inference engine.
pub struct ModelServer {
	pub bus: Arc<RwLock<AugmentBus>>,
	pub loader: Arc<RwLock<ModelLoader>>,
	pub manifest: ConversionStats,
	pub model_dir: PathBuf,
	pub inference: Option<Mutex<InferenceEngine>>,
}

impl ModelServer {
	/// Open a compressed model directory and wire up the augment bus.
	/// Initializes inference engine if config.json + tokenizer.json exist in the directory.
	pub fn open(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
		Self::open_with_progress(model_dir, None)
	}

	/// Open with optional progress tracking for UI integration.
	pub fn open_with_progress(
		model_dir: &Path,
		progress: Option<&crate::inference::progress::LoadingProgress>,
	) -> Result<Self, Box<dyn std::error::Error>> {
		if let Some(p) = progress {
			p.set(1, "Reading manifest...");
		}
		let loader = ModelLoader::open(model_dir)?;
		if let Some(p) = progress {
			p.set(3, "Building tensor map...");
		}
		let manifest = loader.manifest.clone();
		let tensor_names = loader.tensor_names();

		// Build a TensorMap from the manifest for the placement planner
		let mut tensors: Vec<TensorDescriptor> = Vec::with_capacity(tensor_names.len());
		for name in &tensor_names {
			if let Some(stats) = loader.tensor_stats(name) {
				tensors.push(TensorDescriptor {
					name: stats.name.clone(),
					shape: stats.shape.clone(),
					dtype: TensorDtype::F32,
					byte_offset: stats.weight_offset,
					byte_size: stats.original_bytes as u64,
					layer_index: TensorDescriptor::parse_layer_index(name),
				});
			}
		}
		let total_bytes: u64 = tensors.iter().map(|t| t.byte_size).sum();
		let tensor_map = ModelTensorMap {
			model_name: manifest.model_name.clone(),
			tensors,
			total_bytes,
		};

		// Compute weight placement (assume 24GB GPU, 64GB CPU for now)
		let placement_plan = WeightPlacementPlan::plan(&tensor_map, 24_000_000_000, 64_000_000_000);

		// Build augment bus and plug everything in
		let mut bus = AugmentBus::new();
		let model_augment = CompressedModelAugment {
			id: uuid::Uuid::new_v4(),
			model_name: manifest.model_name.clone(),
			weights_path: model_dir.join("weights.bin").to_string_lossy().to_string(),
			sandbag_path: model_dir.join("sandbag.bin").to_string_lossy().to_string(),
			manifest_path: model_dir
				.join("manifest.json")
				.to_string_lossy()
				.to_string(),
			tensor_count: manifest.tensor_count,
			total_core_bytes: manifest.total_core_bytes,
			total_sandbag_bytes: manifest.total_sandbag_bytes,
			total_original_bytes: manifest.total_original_bytes,
			core_ratio: manifest.overall_core_ratio,
			is_active: true,
		};

		bus.plug(Augment::CompressedModel(model_augment), 0);
		bus.plug(Augment::TensorMap(tensor_map), 1);
		bus.plug(Augment::WeightPlan(placement_plan), 2);

		log::info!(
			"ModelServer: {} tensors, core ratio: {:.0}x",
			manifest.tensor_count,
			manifest.overall_core_ratio
		);

		// Initialize inference engine if config.json + tokenizer.json exist
		if let Some(p) = progress {
			p.set(5, "Loading inference engine...");
		}
		let has_config = model_dir.join("config.json").exists();
		let has_tokenizer = model_dir.join("tokenizer.json").exists();
		let inference = if has_config && has_tokenizer {
			match InferenceEngine::open_with_progress(model_dir, progress) {
				Ok(mut engine) => {
					log::info!(
						"ModelServer: inference engine initialized, decompressing to INT4..."
					);
					engine.decompress_all_parallel(8, progress);
					engine.finalize_mmap();
					if let Some(p) = progress {
						p.set(100, "Model ready");
					}
					log::info!("ModelServer: all tensors decompressed and mmap finalized");
					Some(Mutex::new(engine))
				}
				Err(e) => {
					log::warn!(
						"ModelServer: inference engine init failed: {} — running without generation",
						e
					);
					None
				}
			}
		} else {
			log::info!("ModelServer: no config.json/tokenizer.json — tensor access only");
			None
		};

		Ok(Self {
			bus: Arc::new(RwLock::new(bus)),
			loader: Arc::new(RwLock::new(loader)),
			manifest,
			model_dir: model_dir.to_path_buf(),
			inference,
		})
	}

	/// Decompress a single tensor to f32.
	pub fn get_tensor(&self, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
		let mut loader = self.loader.write().unwrap();
		loader.decompress_tensor(name)
	}

	/// Get stats for a tensor.
	pub fn tensor_stats(&self, name: &str) -> Option<TensorStats> {
		let loader = self.loader.read().unwrap();
		loader.tensor_stats(name).cloned()
	}

	/// List all tensor names.
	pub fn tensor_names(&self) -> Vec<String> {
		let loader = self.loader.read().unwrap();
		loader
			.tensor_names()
			.iter()
			.map(|s| s.to_string())
			.collect()
	}

	/// Get the weight placement plan from the bus.
	pub fn placement_plan(&self) -> Option<WeightPlacementPlan> {
		let bus = self.bus.read().unwrap();
		bus.weight_plan().cloned()
	}

	/// Get the compressed model augment from the bus.
	pub fn model_augment(&self) -> Option<CompressedModelAugment> {
		let bus = self.bus.read().unwrap();
		bus.compressed_model().cloned()
	}

	/// Print a summary of all tensors and their placement.
	pub fn print_summary(&self) {
		let plan = self.placement_plan();

		log::info!("\n═══ Model Server Summary ═══");
		log::info!("  Model: {}", self.manifest.model_name);
		log::info!("  Tensors: {}", self.manifest.tensor_count);
		log::info!(
			"  Original: {:.2} GB",
			self.manifest.total_original_bytes as f64 / 1e9
		);
		log::info!(
			"  Core: {:.2} GB (ratio: {:.0}x)",
			self.manifest.total_core_bytes as f64 / 1e9,
			self.manifest.overall_core_ratio
		);
		log::info!(
			"  Sandbag (disk): {:.2} GB",
			self.manifest.total_sandbag_bytes as f64 / 1e9
		);

		if let Some(ref plan) = plan {
			let gpu_count = plan
				.rules
				.iter()
				.filter(|r| r.placement == TensorPlacement::Gpu)
				.count();
			let cpu_count = plan
				.rules
				.iter()
				.filter(|r| r.placement == TensorPlacement::Cpu)
				.count();
			log::info!("  Placement: {} GPU, {} CPU", gpu_count, cpu_count);
		}

		// List first 5 tensors
		log::info!("\n  First tensors:");
		for t in self.manifest.tensors.iter().take(5) {
			log::info!(
				"    {:<45} {:>10} elems  ratio: {:.0}x",
				t.name,
				t.element_count,
				t.core_ratio
			);
		}
		if self.manifest.tensors.len() > 5 {
			log::info!("    ... and {} more", self.manifest.tensors.len() - 5);
		}
	}

	/// Process a chat message through the loaded model.
	/// Returns just the generated text — the [ Agent ] prefix is handled by the messages panel.
	pub fn process_message(&self, msg: &str) -> String {
		log::info!(
			"process_message: received \"{}\"",
			&msg[..msg.len().min(50)]
		);
		if let Some(ref inference) = self.inference {
			let mut engine = inference.lock().unwrap();
			log::info!("process_message: engine locked, calling generate...");
			match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
				engine.generate(msg, 20, 0.7)
			})) {
				Ok(output) if !output.is_empty() => {
					log::info!(
						"process_message: generated {} chars",
						output.chars().count()
					);
					output
				}
				Ok(_) => {
					log::warn!("process_message: empty output");
					"(empty output)".to_string()
				}
				Err(e) => {
					log::error!("process_message: inference panic: {:?}", e);
					"(inference error — check logs)".to_string()
				}
			}
		} else {
			format!(
				"Model not loaded — {} tensors available\n(Set MODEL_DIR to enable generation)",
				self.manifest.tensor_count,
			)
		}
	}
}

/// Materialize a dense model from the compressed seed.
///
/// Does a forward pass through the compressed model, writing each layer's
/// decompressed weights to a new directory. The compressed seed stays
/// unchanged — this spawns an independent dense model.
///
/// The materialization is a side effect of decompression: we read the
/// compressed weights, decompress them, and write the f32 values to disk.
/// No computation (matmul) is needed — just decompress and save.
pub fn materialize_model(
	server: &ModelServer,
	out_dir: &Path,
) -> Result<usize, Box<dyn std::error::Error>> {
	std::fs::create_dir_all(out_dir)?;
	let weights_path = out_dir.join("dense_weights.bin");
	let mut file = std::io::BufWriter::new(std::fs::File::create(&weights_path)?);

	let names = server.tensor_names();
	let total = names.len();
	let mut written = 0usize;

	log::info!("Materializing {} tensors to {}", total, out_dir.display());

	for name in &names {
		log::debug!("Tensor {}/{}: {}", written + 1, total, name);
		let weights = server.get_tensor(name)?;
		let bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();
		file.write_all(&bytes)?;
		written += 1;
	}

	log::info!(
		"Materialized {} tensors to {}",
		written,
		weights_path.display()
	);
	Ok(written)
}

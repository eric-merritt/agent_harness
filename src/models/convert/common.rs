// Shared conversion types, parameters, and weight-compression entry points.
//
// Used by the GGUF, safetensors, and quantization conversion pipelines.

use super::core as core_io;
use super::dedup_count::DedupCountTensor;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorStats {
    pub name: String,
    pub shape: Vec<usize>,
    pub gguf_dtype: u32,
    pub element_count: usize,
    pub original_bytes: usize,
    pub core_bytes: usize,
    pub sandbag_bytes: usize,
    pub core_ratio: f32,
    pub prefix_count: usize,
    pub unique_tail_count: usize,
    pub shared_weights: usize,
    pub mean_precision_lost: f32,
    pub weight_offset: u64,
    pub sandbag_offset: u64,
    pub full_precision: bool,
    #[serde(default)]
    pub quant_offset: u64,
    #[serde(default)]
    pub quant_bytes: usize,
    #[serde(default)]
    pub is_4bit: bool,
    #[serde(default = "default_group_size")]
    pub group_size: usize,
}

fn default_group_size() -> usize { 32 }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConversionStats {
    pub model_name: String,
    pub tensor_count: usize,
    pub total_original_bytes: u64,
    pub total_core_bytes: u64,
    pub total_sandbag_bytes: u64,
    pub overall_core_ratio: f32,
    pub tensors: Vec<TensorStats>,
}

pub const HIGH_PRECISION_TENSORS: &[&str] = &[
    "output.weight",
    "lm_head.weight",
];

pub const HIGH_PRECISION_EXTRA_DIGITS: usize = 0;
pub const HIGH_PRECISION_TRUNCATE_ROUNDS: usize = 0;
pub const CHUNK_SIZE: usize = 1_000_000;
pub const GPU_MIN_ELEMENTS: usize = 100_000;

pub fn resolve_params(name: &str, prefix_digits: usize, truncate_rounds: usize) -> (usize, usize) {
    if HIGH_PRECISION_TENSORS.contains(&name) {
        (prefix_digits + HIGH_PRECISION_EXTRA_DIGITS, HIGH_PRECISION_TRUNCATE_ROUNDS)
    } else {
        (prefix_digits, truncate_rounds)
    }
}

pub struct CompressOutput {
    pub core: Vec<u8>,
    pub sandbag: Vec<u8>,
    pub prefix_count: usize,
    pub unique_tail_count: usize,
    pub shared_weights: usize,
    pub mean_precision_lost: f32,
    pub full_precision: bool,
}

pub fn compress_weights(weights: &[f32], pd: usize, tr: usize) -> CompressOutput {
    if should_be_full_precision(weights) {
        let mut core = Vec::new();
        core.extend_from_slice(&1u32.to_le_bytes());
        for &w in weights {
            core.extend_from_slice(&w.to_le_bytes());
        }
        return CompressOutput {
            core,
            sandbag: Vec::new(),
            prefix_count: 0,
            unique_tail_count: 0,
            shared_weights: 0,
            mean_precision_lost: 0.0,
            full_precision: true,
        };
    }

    if weights.len() > CHUNK_SIZE {
        let n_chunks = (weights.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let mut core = Vec::new();
        core.extend_from_slice(&(n_chunks as u32).to_le_bytes());
        let mut sandbag = Vec::new();
        let mut prefix_count = 0;
        let mut unique_tail_count = 0;
        let mut shared_weights = 0;
        let mut mean_pl_sum = 0.0f32;
        let mut chunk_count = 0u32;

        for chunk in weights.chunks(CHUNK_SIZE) {
            let (t, m) = DedupCountTensor::compress(chunk, pd, tr);
            core.extend(core_io::serialize_core(&t));
            sandbag.extend(m.to_bytes());
            prefix_count += t.prefixes.len();
            unique_tail_count += t.unique_tail_count();
            shared_weights += t.shared_tail_weights();
            mean_pl_sum += t.avg_precision_lost;
            chunk_count += 1;
        }

        return CompressOutput {
            core,
            sandbag,
            prefix_count,
            unique_tail_count,
            shared_weights,
            mean_precision_lost: mean_pl_sum / chunk_count.max(1) as f32,
            full_precision: false,
        };
    } else {
        let (t, m) = DedupCountTensor::compress(weights, pd, tr);
        let core = core_io::serialize_core(&t);
        let sandbag = m.to_bytes();
        let mut full_core = Vec::new();
        full_core.extend_from_slice(&1u32.to_le_bytes());
        full_core.extend(core);

        return CompressOutput {
            core: full_core,
            sandbag,
            prefix_count: t.prefixes.len(),
            unique_tail_count: t.unique_tail_count(),
            shared_weights: t.shared_tail_weights(),
            mean_precision_lost: t.avg_precision_lost,
            full_precision: false,
        };
    }
}

pub fn should_be_full_precision(weights: &[f32]) -> bool {
    let step = (weights.len() / 1000).max(1);
    for i in (0..weights.len()).step_by(step) {
        if weights[i].abs() > 2.0 {
            return true;
        }
    }
    if weights.len() < 8192 {
        return true;
    }
    false
}

pub fn compress_weights_gpu(weights: &[f32], pd: usize, tr: usize) -> Option<CompressOutput> {
    if weights.len() < GPU_MIN_ELEMENTS || weights.len() > CHUNK_SIZE {
        return None;
    }
    let gpu_out = crate::gpu::gpu_compute(weights, pd)?;
    let (t, m) = DedupCountTensor::compress_from_gpu(
        weights, &gpu_out.prefix_bits, &gpu_out.tails, &gpu_out.signs, pd, tr
    );
    let core = core_io::serialize_core(&t);
    let sandbag = m.to_bytes();
    let mut full_core = Vec::new();
    full_core.extend_from_slice(&1u32.to_le_bytes());
    full_core.extend(core);
    Some(CompressOutput {
        core: full_core,
        sandbag,
        prefix_count: t.prefixes.len(),
        unique_tail_count: t.unique_tail_count(),
        shared_weights: t.shared_tail_weights(),
        mean_precision_lost: t.avg_precision_lost,
        full_precision: false,
    })
}

pub fn process_job(
    job: &CompressJob,
    prefix_digits: usize,
    truncate_rounds: usize,
    use_gpu: bool,
) -> CompressResult {
    let n_elems = job.element_count;
    let orig_bytes = n_elems * 4;
    let (pd, tr) = resolve_params(&job.name, prefix_digits, truncate_rounds);

    let out = if use_gpu {
        compress_weights_gpu(&job.weights, pd, tr)
            .unwrap_or_else(|| compress_weights(&job.weights, pd, tr))
    } else {
        compress_weights(&job.weights, pd, tr)
    };

    CompressResult {
        global_idx: job.global_idx,
        stats: TensorStats {
            name: job.name.clone(),
            shape: job.shape.clone(),
            gguf_dtype: 0,
            element_count: n_elems,
            original_bytes: orig_bytes,
            core_bytes: out.core.len(),
            sandbag_bytes: out.sandbag.len(),
            core_ratio: orig_bytes as f32 / out.core.len().max(1) as f32,
            prefix_count: out.prefix_count,
            unique_tail_count: out.unique_tail_count,
            shared_weights: out.shared_weights,
            mean_precision_lost: out.mean_precision_lost,
            weight_offset: 0,
            sandbag_offset: 0,
            full_precision: out.full_precision,
            quant_offset: 0,
            quant_bytes: 0,
            is_4bit: false,
            group_size: 32,
        },
        core: out.core,
        sandbag: out.sandbag,
    }
}

pub struct CompressJob {
    pub global_idx: usize,
    pub name: String,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub weights: Vec<f32>,
}

#[derive(Clone)]
pub struct CompressResult {
    pub global_idx: usize,
    pub stats: TensorStats,
    pub core: Vec<u8>,
    pub sandbag: Vec<u8>,
}

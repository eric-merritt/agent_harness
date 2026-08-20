// GGUF/safetensors → DedupCountTensor conversion.
// Output layout:
//   <out_dir>/
//     manifest.json          — per-tensor metadata (name, shape, dtype, offsets, sizes)
//     weights.bin            — all core compressed data concatenated
//     sandbag.bin               — all per-weight metadata concatenated

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufWriter, Write, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use super::dedup_count::{DedupCountTensor, Sandbag, UniqueTail, DataFlag};
use super::gguf::GGUFFile;
use super::safetensors::{SafetensorsHeader, SafetensorsDtype};

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionStats {
    pub model_name: String,
    pub tensor_count: usize,
    pub total_original_bytes: u64,
    pub total_core_bytes: u64,
    pub total_sandbag_bytes: u64,
    pub overall_core_ratio: f32,
    pub tensors: Vec<TensorStats>,
}

const HIGH_PRECISION_TENSORS: &[&str] = &[
    "output.weight",
    "lm_head.weight",
];

const HIGH_PRECISION_EXTRA_DIGITS: usize = 0;
const HIGH_PRECISION_TRUNCATE_ROUNDS: usize = 0;
const CHUNK_SIZE: usize = 1_000_000;
const GPU_MIN_ELEMENTS: usize = 100_000;

pub fn normalize_tensor_name(name: &str) -> String {
    if name == "model.embed_tokens.weight" || name == "embed_tokens.weight" {
        return "token_embd.weight".to_string();
    }
    if name == "model.norm.weight" || name == "norm.weight" {
        return "output_norm.weight".to_string();
    }
    if name == "lm_head.weight" || name == "model.lm_head.weight" {
        return "output.weight".to_string();
    }

    if let Some(rest) = name.strip_prefix("model.layers.") {
        if let Some(dot) = rest.find('.') {
            let layer_num = &rest[..dot];
            let suffix = &rest[dot + 1..];
            let gguf_name = match suffix {
                "input_layernorm.weight" => Some("attn_norm.weight"),
                "post_attention_layernorm.weight" => Some("post_attention_norm.weight"),
                "self_attn.q_proj.weight" => Some("attn_q.weight"),
                "self_attn.k_proj.weight" => Some("attn_k.weight"),
                "self_attn.v_proj.weight" => Some("attn_v.weight"),
                "self_attn.o_proj.weight" => Some("attn_output.weight"),
                "self_attn.q_norm.weight" => Some("attn_q_norm.weight"),
                "self_attn.k_norm.weight" => Some("attn_k_norm.weight"),
                "mlp.gate_proj.weight" => Some("ffn_gate.weight"),
                "mlp.up_proj.weight" => Some("ffn_up.weight"),
                "mlp.down_proj.weight" => Some("ffn_down.weight"),
                _ => None,
            };
            if let Some(gguf) = gguf_name {
                return format!("blk.{}.{}", layer_num, gguf);
            }
            return format!("blk.{}.{}", layer_num, suffix);
        }
    }
    name.to_string()
}

struct CompressOutput {
    core: Vec<u8>,
    sandbag: Vec<u8>,
    prefix_count: usize,
    unique_tail_count: usize,
    shared_weights: usize,
    mean_precision_lost: f32,
    full_precision: bool,
}

fn compress_weights(weights: &[f32], pd: usize, tr: usize) -> CompressOutput {
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
            core.extend(serialize_core(&t));
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
        let core = serialize_core(&t);
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

fn should_be_full_precision(weights: &[f32]) -> bool {
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

fn resolve_params(name: &str, prefix_digits: usize, truncate_rounds: usize) -> (usize, usize) {
    if HIGH_PRECISION_TENSORS.contains(&name) {
        (prefix_digits + HIGH_PRECISION_EXTRA_DIGITS, HIGH_PRECISION_TRUNCATE_ROUNDS)
    } else {
        (prefix_digits, truncate_rounds)
    }
}


fn compress_weights_gpu(weights: &[f32], pd: usize, tr: usize) -> Option<CompressOutput> {
    if weights.len() < GPU_MIN_ELEMENTS || weights.len() > CHUNK_SIZE {
        return None;
    }
    let gpu_out = crate::gpu::gpu_compute(weights, pd)?;
    let (t, m) = DedupCountTensor::compress_from_gpu(
        weights, &gpu_out.prefix_bits, &gpu_out.tails, &gpu_out.signs, pd, tr
    );
    let core = serialize_core(&t);
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

fn process_job(job: &CompressJob, prefix_digits: usize, truncate_rounds: usize, use_gpu: bool) -> CompressResult {
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

fn serialize_core(tensor: &DedupCountTensor) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&(tensor.count as u32).to_le_bytes());
    data.extend_from_slice(&(tensor.prefixes.len() as u32).to_le_bytes());
    data.extend_from_slice(&(tensor.unique_tails.len() as u32).to_le_bytes());
    data.extend_from_slice(&(tensor.prefix_digits as u32).to_le_bytes());
    data.extend_from_slice(&(tensor.tail_digits as u32).to_le_bytes());
    data.extend_from_slice(&tensor.avg_precision_lost.to_le_bytes());

    for &p in &tensor.prefixes {
        data.extend_from_slice(&p.to_le_bytes());
    }
    for ut in &tensor.unique_tails {
        data.extend_from_slice(&ut.value.to_le_bytes());
    }
    data.push(DataFlag::GapFlag as u8);
    for ut in tensor.unique_tails.iter().rev() {
        data.extend_from_slice(&ut.repeat_count.to_le_bytes());
    }
    data.push(DataFlag::TailFlag as u8);
    for &pc in tensor.prefix_counts.iter().rev() {
        data.extend_from_slice(&pc.to_le_bytes());
    }
    data.push(DataFlag::CountFlag as u8);
    data
}

fn deserialize_core(data: &[u8]) -> Option<DedupCountTensor> {
    let mut pos = 0;
    deserialize_core_at(data, &mut pos)
}

pub fn deserialize_core_at(data: &[u8], pos: &mut usize) -> Option<DedupCountTensor> {
    if data.len() < *pos + 24 { return None; }
    let count = u32::from_le_bytes(data[*pos..*pos+4].try_into().ok()?) as usize;
    let prefix_count = u32::from_le_bytes(data[*pos+4..*pos+8].try_into().ok()?) as usize;
    let tail_count = u32::from_le_bytes(data[*pos+8..*pos+12].try_into().ok()?) as usize;
    let prefix_digits = u32::from_le_bytes(data[*pos+12..*pos+16].try_into().ok()?) as usize;
    let tail_digits = u32::from_le_bytes(data[*pos+16..*pos+20].try_into().ok()?) as usize;
    let avg_precision_lost = f32::from_le_bytes(data[*pos+20..*pos+24].try_into().ok()?);
    *pos += 24;

    let mut prefixes = Vec::with_capacity(prefix_count);
    for _ in 0..prefix_count {
        if data.len() < *pos + 2 { return None; }
        prefixes.push(u16::from_le_bytes(data[*pos..*pos+2].try_into().ok()?));
        *pos += 2;
    }
    
    let mut unique_tails = Vec::with_capacity(tail_count);
    for _ in 0..tail_count {
        if data.len() < *pos + 2 { return None; }
        let value = u16::from_le_bytes(data[*pos..*pos+2].try_into().ok()?);
        unique_tails.push(UniqueTail { value, repeat_count: 0 });
        *pos += 2;
    }
    
    if *pos < data.len() && data[*pos] == DataFlag::GapFlag as u8 { *pos += 1; }

    let tail_counts_bytes = tail_count * 4;
    let prefix_counts_bytes = prefix_count * 4;
    let total_rear_bytes = tail_counts_bytes + 1 + prefix_counts_bytes + 1;
    if data.len() < *pos + total_rear_bytes { return None; }

    let tail_counts_start = *pos;
    let tail_flag_pos = tail_counts_start + tail_counts_bytes;
    let prefix_counts_start = tail_flag_pos + 1;
    let count_flag_pos = prefix_counts_start + prefix_counts_bytes;

    if data[tail_flag_pos] != DataFlag::TailFlag as u8 { return None; }
    if data[count_flag_pos] != DataFlag::CountFlag as u8 { return None; }

    for (i, ut) in unique_tails.iter_mut().rev().enumerate() {
        let chunk_offset = tail_counts_start + (i * 4);
        ut.repeat_count = u32::from_le_bytes(data[chunk_offset..chunk_offset + 4].try_into().ok()?);
    }

    let mut prefix_counts = vec![0u32; prefix_count];
    for i in 0..prefix_count {
        let chunk_offset = prefix_counts_start + (i * 4);
        prefix_counts[prefix_count - 1 - i] = u32::from_le_bytes(data[chunk_offset..chunk_offset + 4].try_into().ok()?);
    }

    *pos = count_flag_pos + 1;

    Some(DedupCountTensor {
        count,
        prefixes,
        unique_tails,
        prefix_counts,
        prefix_digits,
        tail_digits,
        avg_precision_lost,
    })
}

pub fn deserialize_core_chunks(data: &[u8]) -> Vec<DedupCountTensor> {
    if data.len() < 4 { return Vec::new(); }
    let chunk_count = u32::from_le_bytes(data[0..4].try_into().unwrap_or([0,0,0,0])) as usize;
    let mut pos = 4;
    let mut chunks = Vec::with_capacity(chunk_count);
    for _ in 0..chunk_count {
        if let Some(t) = deserialize_core_at(data, &mut pos) {
            chunks.push(t);
        }
    }
    chunks
}
pub fn convert_gguf(
    gguf_path: &Path,
    out_dir: &Path,
    prefix_digits: usize,
    truncate_rounds: usize,
    num_workers: usize,
) -> Result<ConversionStats, Box<dyn std::error::Error>> {
    let gguf = GGUFFile::from_file(gguf_path)?;
    let model_name = gguf.model_name().to_string();
    let tensor_count = gguf.tensor_info.len();
    log::info!("Blocks Found: {}", tensor_count);

    let mut tensor_plan: Vec<(usize, usize, u32, usize)> = Vec::new();
    for i in 0..tensor_count {
        let info = &gguf.tensor_info[i];
        let n_elems = info.element_count() as usize;
        let dtype = info.dtype;
        if super::gguf::quant_block_info(dtype).is_none() && dtype != 0 && dtype != 1 && dtype != 30 {
            continue;
        }
        if n_elems == 0 { continue; }
        let (block_elems, block_bytes) = super::gguf::quant_block_info(dtype).unwrap_or((1, 4));
        let blocks_per_chunk = (CHUNK_SIZE + block_elems - 1) / block_elems;
        let chunk_bytes = blocks_per_chunk * block_bytes;
        let total_bytes = info.byte_size() as usize;
        let n_chunks = ((total_bytes + chunk_bytes - 1) / chunk_bytes).max(1);
        tensor_plan.push((i, n_elems, dtype, n_chunks));
    }

    let mut tensor_base: Vec<usize> = Vec::with_capacity(tensor_plan.len());
    let mut acc = 0usize;
    for (_, _, _, nchunks) in &tensor_plan {
        tensor_base.push(acc);
        acc += *nchunks;
    }
    let total_chunks = acc;
    log::info!("Total chunks: {} across {} workers", total_chunks, num_workers);

    fs::create_dir_all(out_dir)?;
    let gguf_parent = gguf_path.parent().unwrap_or(Path::new("/"));
    if out_dir.canonicalize().ok() == gguf_parent.canonicalize().ok() {
        return Err("Output directory must not be the same as the input GGUF directory".into());
    }
    let weights_path = out_dir.join("weights.bin");
    let sandbag_path = out_dir.join("sandbag.bin");
    let manifest_path = out_dir.join("manifest.json");

    let (job_tx, job_rx) = mpsc::sync_channel::<CompressJob>(4);
    let job_rx = Arc::new(Mutex::new(job_rx));
    let (result_tx, result_rx) = mpsc::channel::<CompressResult>();

    let gpu_loaders = 7;
    let (gpu_job_tx, gpu_job_tx_internal) = mpsc::sync_channel::<CompressJob>(gpu_loaders + 1);
    let gpu_job_rx = Arc::new(Mutex::new(gpu_job_tx_internal));

    let loader_handles: Vec<_> = (0..gpu_loaders).map(|_| {
        let job_rx = Arc::clone(&job_rx);
        let gpu_job_tx = gpu_job_tx.clone();
        thread::spawn(move || {
            loop {
                let job = {
                    let rx = job_rx.lock().unwrap();
                    rx.recv()
                };
                let job = match job { Ok(j) => j, Err(_) => break };
                if gpu_job_tx.send(job).is_err() { break; }
            }
        })
    }).collect();
    drop(gpu_job_tx);
    let gpu_handle = {
        let gpu_job_rx = Arc::clone(&gpu_job_rx);
        let result_tx = result_tx.clone();
        thread::spawn(move || {
            loop {
                let job = {
                    let rx = gpu_job_rx.lock().unwrap();
                    rx.recv()
                };
                let job = match job { Ok(j) => j, Err(_) => break };
                let _ = result_tx.send(process_job(&job, prefix_digits, truncate_rounds, true));
            }
        })
    };

    let cpu_workers = num_workers.saturating_sub(gpu_loaders + 1);
    let worker_handles: Vec<_> = (0..cpu_workers).map(|_| {
        let job_rx = Arc::clone(&job_rx);
        let result_tx = result_tx.clone();
        thread::spawn(move || {
            loop {
                let job = {
                    let rx = job_rx.lock().unwrap();
                    rx.recv()
                };
                let job = match job { Ok(j) => j, Err(_) => break };
                let _ = result_tx.send(process_job(&job, prefix_digits, truncate_rounds, false));
            }
        })
    }).collect();
    drop(result_tx);

    let gguf2 = GGUFFile::from_file(gguf_path)?;
    let tensor_plan2 = tensor_plan.clone();
    let tensor_base2 = tensor_base.clone();
    let gguf_path_owned = gguf_path.to_path_buf();
    
    let reader = thread::spawn(move || -> Result<(), String> {
        let mut file = File::open(&gguf_path_owned).map_err(|e| e.to_string())?;
        for (tpidx, (tidx, n_elems, dtype, nchunks)) in tensor_plan2.iter().enumerate() {
            let info = &gguf2.tensor_info[*tidx];
            let (block_elems, block_bytes) = super::gguf::quant_block_info(*dtype).unwrap_or((1, 4));
            let blocks_per_chunk = (CHUNK_SIZE + block_elems - 1) / block_elems;
            let chunk_bytes = blocks_per_chunk * block_bytes;
            let total_bytes = info.byte_size() as usize;
            
            for c in 0..*nchunks {
                let byte_offset = c * chunk_bytes;
                let read_len = chunk_bytes.min(total_bytes.saturating_sub(byte_offset));
                if read_len == 0 { continue; }
                let raw_chunk = gguf2.read_tensor_range(&mut file, *tidx, byte_offset, read_len)
                    .map_err(|e| e.to_string())?;
                let chunk_n = (raw_chunk.len() / block_bytes) * block_elems;
                let weights = gguf2.dequantize_to_f32(&raw_chunk, *dtype, chunk_n.max(*n_elems));                
                let base_idx = tensor_base2[tpidx];
                let job = CompressJob {
                    global_idx: base_idx + c,
                    name: info.name.clone(),
                    shape: info.dim.iter().map(|d| *d as usize).collect(),
                    element_count: weights.len(),
                    weights,
                };
                if job_tx.send(job).is_err() { break; }
            }
        }
        Ok(())
    });

    let mut weights_file = BufWriter::new(File::create(&weights_path)?);
    let mut sandbag_file = BufWriter::new(File::create(&sandbag_path)?);
    let mut tensor_chunk_stats: Vec<Vec<TensorStats>> = vec![Vec::new(); tensor_plan.len()];
    let mut buffer: HashMap<usize, (Vec<u8>, Vec<u8>, TensorStats)> = HashMap::new();
    let mut expected_chunk = 0usize;

    let mut all_stats: Vec<TensorStats> = Vec::new();
    let mut t_orig: u64 = 0;
    let mut t_core: u64 = 0;
    let mut t_meta: u64 = 0;

    let chunk_lookup: Vec<(usize, usize)> = {
        let mut v = Vec::with_capacity(total_chunks);
        for (tpidx, (_, _, _, nchunks)) in tensor_plan.iter().enumerate() {
            for c in 0..*nchunks {
                v.push((tpidx, c));
            }
        }
        v
    };

    while expected_chunk < total_chunks {
        let result = match result_rx.recv() {
            Ok(r) => r,
            Err(_) => return Err("worker channel closed unexpectedly".into()),
        };
        buffer.insert(result.global_idx, (result.core, result.sandbag, result.stats));

        while let Some((core, sandbag, stats)) = buffer.remove(&expected_chunk) {
            weights_file.write_all(&core)?;
            sandbag_file.write_all(&sandbag)?;
            let (tpidx, _c) = chunk_lookup[expected_chunk];
            tensor_chunk_stats[tpidx].push(stats);
            expected_chunk += 1;
        }
    }

    log::info!("Assembling manifest...");
    let mut running_woff = 0u64;
    let mut running_moff = 0u64;
    for (tpidx, (tidx, n_elems, dtype, nchunks)) in tensor_plan.iter().enumerate() {
        let info = &gguf.tensor_info[*tidx];
        let chunks = &tensor_chunk_stats[tpidx];
        let tc_bytes: usize = chunks.iter().map(|s| s.core_bytes).sum();
        let ts_bytes: usize = chunks.iter().map(|s| s.sandbag_bytes).sum();
        let tp_count: usize = chunks.iter().map(|s| s.prefix_count).sum();
        let tut_count: usize = chunks.iter().map(|s| s.unique_tail_count).sum();
        let tsw: usize = chunks.iter().map(|s| s.shared_weights).sum();
        let tmpl: f32 = chunks.iter().map(|s| s.mean_precision_lost).sum::<f32>() / *nchunks as f32;
        let is_fp = chunks.iter().any(|s| s.full_precision);
        
        let tensor_woff = running_woff;
        let tensor_moff = running_moff;
        running_woff += tc_bytes as u64;
        running_moff += ts_bytes as u64;
        let orig_bytes = n_elems * 4;

        all_stats.push(TensorStats {
            name: info.name.clone(),
            shape: info.dim.iter().map(|d| *d as usize).collect(),
            gguf_dtype: *dtype,
            element_count: *n_elems,
            original_bytes: orig_bytes,
            core_bytes: tc_bytes,
            sandbag_bytes: ts_bytes,
            core_ratio: orig_bytes as f32 / tc_bytes.max(1) as f32,
            prefix_count: tp_count,
            unique_tail_count: tut_count,
            shared_weights: tsw,
            mean_precision_lost: tmpl,
            weight_offset: tensor_woff,
            sandbag_offset: tensor_moff,
            full_precision: is_fp,
            quant_offset: 0, quant_bytes: 0, is_4bit: false, group_size: 32,
        });
        t_orig += orig_bytes as u64;
        t_core += tc_bytes as u64;
        t_meta += ts_bytes as u64;
    }

    let _ = gpu_handle.join();
    for h in loader_handles { let _ = h.join(); }
    let _ = reader.join();
    for h in worker_handles { let _ = h.join(); }

    let overall_ratio = if t_core > 0 { t_orig as f32 / t_core as f32 } else { 1.0 };
    let conv_stats = ConversionStats {
        model_name, tensor_count: all_stats.len(),
        total_original_bytes: t_orig, total_core_bytes: t_core,
        total_sandbag_bytes: t_meta, overall_core_ratio: overall_ratio, tensors: all_stats,
    };
    fs::write(&manifest_path, serde_json::to_string_pretty(&conv_stats)?)?;
    conv_stats.to_binary_cache(out_dir)?;

    let config = crate::inference::config::ModelConfig::from_gguf(&gguf);
    config.to_file(out_dir)?;
    crate::inference::tokenizer::Tokenizer::extract_to_file(&gguf, out_dir)?;
    Ok(conv_stats)
}

impl ConversionStats {
    fn to_binary_cache(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let bin_path = dir.join("manifest.bin");
        if let Ok(data) = bincode::serialize(self) {
            let _ = fs::write(bin_path, data);
        }
        Ok(())
    }
}

pub struct ModelLoader {
    pub manifest: ConversionStats,
    weights_file: File,
    sandbag_file: File,
    tensor_index: HashMap<String, usize>,
    pub dir: PathBuf,
}

impl ModelLoader {
    pub fn open(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let manifest_path = dir.join("manifest.json");
        let weights_path = dir.join("weights.bin");
        let sandbag_path = dir.join("sandbag.bin");
        let bin_path = dir.join("manifest.bin");

        let manifest: ConversionStats = if bin_path.exists() {
            let data = fs::read(&bin_path)?;
            bincode::deserialize::<ConversionStats>(&data).unwrap_or_else(|_| {
                let manifest_str = fs::read_to_string(&manifest_path).unwrap_or_default();
                serde_json::from_str(&manifest_str).unwrap()
            })
        } else {
            let manifest_str = fs::read_to_string(&manifest_path)?;
            serde_json::from_str(&manifest_str)?
        };

        let mut tensor_index = HashMap::new();
        for (i, t) in manifest.tensors.iter().enumerate() {
            tensor_index.insert(t.name.clone(), i);
        }

        let weights_file = File::open(&weights_path)?;
        let sandbag_file = File::open(&sandbag_path)?;

        Ok(Self { manifest, weights_file, sandbag_file, tensor_index, dir: dir.to_path_buf() })
    }

    /// Get stats for a specific tensor by name.
    pub fn tensor_stats(&self, name: &str) -> Option<&TensorStats> {
        self.tensor_index.get(name)
            .and_then(|&i| self.manifest.tensors.get(i))
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.manifest.tensors.iter()
            .map(|t| t.name.as_str())
            .collect()
    }

    pub fn read_tensor_raw(&mut self, name: &str) -> Result<(Vec<u8>, Vec<u8>, TensorStats), Box<dyn std::error::Error>> {
        let idx = *self.tensor_index.get(name).ok_or_else(|| format!("tensor not found: {}", name))?;
        let stats = self.manifest.tensors[idx].clone();

        self.weights_file.seek(SeekFrom::Start(stats.weight_offset))?;
        let mut core_buf = vec![0u8; stats.core_bytes];
        self.weights_file.read_exact(&mut core_buf)?;

        self.sandbag_file.seek(SeekFrom::Start(stats.sandbag_offset))?;
        let mut sandbag_buf = vec![0u8; stats.sandbag_bytes];
        self.sandbag_file.read_exact(&mut sandbag_buf)?;

        Ok((core_buf, sandbag_buf, stats))
    }

    pub fn decompress_tensor(&mut self, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.decompress_tensor_impl(name, None, true)
    }

    pub fn decompress_tensor_global(&mut self, name: &str, global: Arc<crate::models::dedup_count::GlobalTable>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.decompress_tensor_impl(name, Some(global), true)
    }

    pub fn decompress_tensor_global_single(&mut self, name: &str, global: Arc<crate::models::dedup_count::GlobalTable>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.decompress_tensor_impl(name, Some(global), false)
    }
}
struct ChunkInfo {
    core_pos: usize,
    sand_pos: usize,
    element_count: usize,
    remap: Option<crate::models::dedup_count::ChunkRemap>,
}

impl ModelLoader {
    fn decompress_tensor_impl(
        &mut self, name: &str, global: Option<Arc<crate::models::dedup_count::GlobalTable>>, parallel_chunks: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
           let idx = *self.tensor_index.get(name)
            .ok_or_else(|| format!("tensor not found: {}", name))?;
        
        // 1. Copy the fields we need immediately so we can drop the borrow on self
        let is_full_precision = self.manifest.tensors[idx].full_precision;
        let sandbag_bytes = self.manifest.tensors[idx].sandbag_bytes;
        let core_bytes = self.manifest.tensors[idx].core_bytes;
        let weight_offset = self.manifest.tensors[idx].weight_offset;
        let element_count = self.manifest.tensors[idx].element_count;

        // 2. Full-precision path: Handle using the copied metadata
        if is_full_precision || sandbag_bytes == 0 {
            self.weights_file.seek(SeekFrom::Start(weight_offset))?;
            let mut buf = vec![0u8; core_bytes];
            self.weights_file.read_exact(&mut buf)?;
            let weights: Vec<f32> = buf[4..].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            return Ok(weights);
        }

        // 3. Compressed path: Call read_tensor_raw safely now that self is unborrowed
        let start_decompress = std::time::Instant::now();
        
        // CRITICAL FIX: Destructure the tuple directly from the Result returned by read_tensor_raw
        let (core_buf, sandbag_buf, stats) = self.read_tensor_raw(name)?;

        let mut chunks: Vec<ChunkInfo> = Vec::new();
        let mut core_pos = 4usize;
        let mut sand_pos = 0usize;

        let n_chunks = u32::from_le_bytes(core_buf[0..4].try_into().unwrap()) as usize;

        for _ in 0..n_chunks {
            if core_pos >= core_buf.len() { break; }
            let chunk_core_pos = core_pos;
            let chunk_sand_pos = sand_pos;

            let tensor = match deserialize_core_at(&core_buf, &mut core_pos) {
                Some(t) => t,
                None => break,
            };
            let chunk_sand = match Sandbag::from_bytes(&sandbag_buf[sand_pos..]) {
                Some(s) => s,
                None => break,
            };
            sand_pos += chunk_sand.bytes();
            let remap = global.as_ref().map(|gt| gt.build_chunk_remap(&tensor));

            chunks.push(ChunkInfo { core_pos: chunk_core_pos, sand_pos: chunk_sand_pos, element_count: tensor.count, remap });
        }

        let mut offsets = vec![0usize; chunks.len()];
        let mut offset = 0usize;
        for (i, ci) in chunks.iter().enumerate() {
            offsets[i] = offset;
            offset += ci.element_count;
        }

        let core_data = Arc::new(core_buf);
        let sand_data = Arc::new(sandbag_buf);
        let chunks_data = Arc::new(chunks);
        let offsets_data = Arc::new(offsets);
        let mut result = vec![0.0f32; stats.element_count];
        let mut total_written = 0usize;
        let mut worker_errors = 0usize;

        if parallel_chunks && chunks_data.len() > 1 {
            let work_queue = Arc::new(Mutex::new((0..chunks_data.len()).collect::<VecDeque<usize>>()));
            let (result_tx, result_rx) = mpsc::channel::<(usize, Vec<f32>)>();
            let num_workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(32);

            let handles: Vec<_> = (0..num_workers).map(|_| {
                let wq = Arc::clone(&work_queue);
                let core = Arc::clone(&core_data);
                let sand = Arc::clone(&sand_data);
                let chunks_ref = Arc::clone(&chunks_data);
                let offsets_ref = Arc::clone(&offsets_data);
                let tx = result_tx.clone();
                let gt_cloned = global.clone();

                thread::spawn(move || {
                    loop {
                        let idx = { wq.lock().unwrap().pop_front() };
                        let Some(idx) = idx else { break; };
                        let ci = &chunks_ref[idx];
                        let mut cp = ci.core_pos;
                        let tensor = match deserialize_core_at(&core, &mut cp) { Some(t) => t, None => continue };
                        let chunk_sand = match Sandbag::from_bytes(&sand[ci.sand_pos..]) { Some(s) => s, None => continue };
                        
                        let decompressed = if let Some(ref gt) = gt_cloned {
                            if let Some(ref remap) = ci.remap { gt.decompress_with_remap(&chunk_sand, &tensor, remap) }
                            else { tensor.decompress_all_global(&chunk_sand, gt) }
                        } else { tensor.decompress_all(&chunk_sand) };

                        let _ = tx.send((offsets_ref[idx], decompressed));
                    }
                })
            }).collect();
            drop(result_tx);

            for (offset, data) in result_rx {
                let end = offset + data.len();
                if end <= result.len() {
                    result[offset..end].copy_from_slice(&data);
                    total_written += data.len();
                } else { worker_errors += 1; }
            }
            for h in handles { if h.join().is_err() { worker_errors += 1; } }
        } else {
            // ... (Sequential block flows gracefully into the next file chunk)
            for (idx, ci) in chunks_data.iter().enumerate() {
                let mut cp = ci.core_pos;
                let tensor = match deserialize_core_at(&core_data, &mut cp) {
                    Some(t) => t,
                    None => { log::error!("chunk header vanished for {} chunk {}", name, idx); continue; }
                };
                let chunk_sand = match Sandbag::from_bytes(&sand_data[ci.sand_pos..]) {
                    Some(s) => s,
                    None => { log::error!("chunk sandbag vanished for {} chunk {}", name, idx); continue; }
                };
                let decompressed = if let Some(ref gt) = global.as_ref() {
                    if let Some(ref remap) = ci.remap { gt.decompress_with_remap(&chunk_sand, &tensor, remap) }
                    else { tensor.decompress_all_global(&chunk_sand, gt) }
                } else { tensor.decompress_all(&chunk_sand) };

                let offset = offsets_data[idx];
                let end = offset + decompressed.len();
                if end <= result.len() {
                    result[offset..end].copy_from_slice(&decompressed);
                    total_written += decompressed.len();
                } else { worker_errors += 1; }
            }
        }

        log::debug!("Decompressed {} in {:.2}s", name, start_decompress.elapsed().as_secs_f64());
        if total_written != stats.element_count {
            log::error!("decompressed {}/{} elements for {} ({} errors)", total_written, stats.element_count, name, worker_errors);
        }
        Ok(result)
    }

    pub fn to_augment(&self) -> crate::augment::augment::CompressedModelAugment {
        crate::augment::augment::CompressedModelAugment {
            id: Uuid::new_v4(),
            model_name: self.manifest.model_name.clone(),
            weights_path: self.dir.join("weights.bin").to_string_lossy().to_string(),
            sandbag_path: self.dir.join("sandbag.bin").to_string_lossy().to_string(),
            manifest_path: self.dir.join("manifest.json").to_string_lossy().to_string(),
            tensor_count: self.manifest.tensor_count,
            total_core_bytes: self.manifest.total_core_bytes,
            total_sandbag_bytes: self.manifest.total_sandbag_bytes,
            total_original_bytes: self.manifest.total_original_bytes,
            core_ratio: self.manifest.overall_core_ratio,
            is_active: true,
        }
    }
}
pub fn convert_safetensors(
    safetensors_path: &Path,
    out_dir: &Path,
    prefix_digits: usize,
    truncate_rounds: usize,
) -> Result<ConversionStats, Box<dyn std::error::Error>> {
    let data = fs::read(safetensors_path)?;
    let (header, data_start) = SafetensorsHeader::parse_from_bytes(&data)?;
    let model_name = safetensors_path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
    let tensor_count = header.tensor_info.len();
    log::info!("Converting safetensors: {} ({} tensors)", model_name, tensor_count);

    fs::create_dir_all(out_dir)?;
    let st_parent = safetensors_path.parent().unwrap_or(Path::new("/"));
    if out_dir.canonicalize().ok() == st_parent.canonicalize().ok() {
        return Err("Output directory must not be the same as the input directory".into());
    }
    let weights_path = out_dir.join("weights.bin");
    let sandbag_path = out_dir.join("sandbag.bin");
    let manifest_path = out_dir.join("manifest.json");

    let mut weights_file = BufWriter::new(File::create(&weights_path)?);
    let mut sandbag_file = BufWriter::new(File::create(&sandbag_path)?);
    let mut all_stats: Vec<TensorStats> = Vec::new();
    let mut weight_offset: u64 = 0;
    let mut sandbag_offset: u64 = 0;
    let mut total_original: u64 = 0;
    let mut total_core: u64 = 0;
    let mut total_meta: u64 = 0;

    let sorted = header.sorted_tensors();
    for (i, (name, info)) in sorted.into_iter().enumerate() {
        let n_elems = info.element_count() as usize;
        if n_elems == 0 { continue; }

        let raw = &data[data_start + info.data_offset()..][..info.byte_size()];
        let weights = info.dtype.dequantize_to_f32(raw, n_elems);

        let (pd, tr) = resolve_params(&name, prefix_digits, truncate_rounds);
        let out = compress_weights(&weights, pd, tr);
        let orig_bytes = n_elems * 4;
        let core_len = out.core.len() as u64;
        let sandbag_len = out.sandbag.len() as u64;

        weights_file.write_all(&out.core)?;
        sandbag_file.write_all(&out.sandbag)?;

        all_stats.push(TensorStats {
            name: name.clone(),
            shape: info.shape.clone(),
            gguf_dtype: 0,
            element_count: n_elems,
            original_bytes: orig_bytes,
            core_bytes: out.core.len(),
            sandbag_bytes: out.sandbag.len(),
            core_ratio: orig_bytes as f32 / core_len.max(1) as f32,
            prefix_count: out.prefix_count,
            unique_tail_count: out.unique_tail_count,
            shared_weights: out.shared_weights,
            mean_precision_lost: out.mean_precision_lost,
            weight_offset,
            sandbag_offset,
            full_precision: out.full_precision,
            quant_offset: 0, quant_bytes: 0, is_4bit: false, group_size: 32,
        });

        weight_offset += core_len;
        sandbag_offset += sandbag_len;
        total_original += orig_bytes as u64;
        total_core += core_len;
        total_meta += sandbag_len;
    }

    drop(weights_file);
    drop(sandbag_file);

    let overall_ratio = if total_core > 0 { total_original as f32 / total_core as f32 } else { 1.0 };
    let stats = ConversionStats {
        model_name, tensor_count: all_stats.len(),
        total_original_bytes: total_original, total_core_bytes: total_core,
        total_sandbag_bytes: total_meta, overall_core_ratio: overall_ratio, tensors: all_stats,
    };
    fs::write(&manifest_path, serde_json::to_string_pretty(&stats)?)?;
    stats.to_binary_cache(out_dir)?;
    Ok(stats)
}

struct CompressJob {
    global_idx: usize,
    name: String,
    shape: Vec<usize>,
    element_count: usize,
    weights: Vec<f32>,
}

impl CompressJob {
    fn high_precision(&self) -> bool { HIGH_PRECISION_TENSORS.contains(&self.name.as_str()) }
}

#[derive(Clone)]
struct CompressResult {
    global_idx: usize,
    stats: TensorStats,
    core: Vec<u8>,
    sandbag: Vec<u8>,
}

pub fn convert_safetensors_parallel(
    shard_paths: &[PathBuf],
    out_dir: &Path,
    prefix_digits: usize,
    truncate_rounds: usize,
    num_workers: usize,
) -> Result<ConversionStats, Box<dyn std::error::Error>> {
    let mut shard_info: Vec<(PathBuf, SafetensorsHeader, usize)> = Vec::new();
    for path in shard_paths {
        let (header, data_start) = SafetensorsHeader::parse_from_file(path)?;
        shard_info.push((path.clone(), header, data_start));
    }

    let global_tensors: Vec<(usize, usize, String, SafetensorsDtype, Vec<usize>, usize, usize, usize)> = {
        let mut v = Vec::new();
        for (shard_idx, (_, header, data_start)) in shard_info.iter().enumerate() {
            for (name, info) in header.sorted_tensors() {
                if name.ends_with(".bias") { continue; }
                let normalized = normalize_tensor_name(name);
                v.push((v.len(), shard_idx, normalized, info.dtype, info.shape.clone(), info.data_offset(), info.byte_size(), *data_start));
            }
        }
        v
    };

    let total_tensors = global_tensors.len();
    fs::create_dir_all(out_dir)?;

    let (job_tx, job_rx) = mpsc::sync_channel::<CompressJob>(4);
    let job_rx = Arc::new(Mutex::new(job_rx));
    let (result_tx, result_rx) = mpsc::channel::<CompressResult>();

    let worker_handles: Vec<_> = (0..num_workers).map(|_| {
        let job_rx = Arc::clone(&job_rx);
        let result_tx = result_tx.clone();
        thread::spawn(move || {
            loop {
                let job = { let rx = job_rx.lock().unwrap(); rx.recv() };
                let job = match job { Ok(j) => j, Err(_) => break };
                let (pd, tr) = resolve_params(&job.name, prefix_digits, truncate_rounds);
                let out = compress_weights(&job.weights, pd, tr);

                let _ = result_tx.send(CompressResult {
                    global_idx: job.global_idx,
                    stats: TensorStats {
                        name: job.name.clone(), shape: job.shape.clone(), gguf_dtype: 0,
                        element_count: job.element_count, original_bytes: job.element_count * 4,
                        core_bytes: out.core.len(), sandbag_bytes: out.sandbag.len(),
                        core_ratio: (job.element_count * 4) as f32 / out.core.len().max(1) as f32,
                        prefix_count: out.prefix_count, unique_tail_count: out.unique_tail_count,
                        shared_weights: out.shared_weights, mean_precision_lost: out.mean_precision_lost,
                        weight_offset: 0, sandbag_offset: 0, full_precision: out.full_precision,
                        quant_offset: 0, quant_bytes: 0, is_4bit: false, group_size: 32,
                    },
                    core: out.core, sandbag: out.sandbag,
                });
            }
        })
    }).collect();
    drop(result_tx);
    let num_shards = shard_paths.len();
    let spr = (num_shards + 1) / 2;
    let reader_handles: Vec<_> = (0..2).filter_map(|rid| {
        let rs = rid * spr;
        if rs >= num_shards { return None; }
        let re = (rs + spr).min(num_shards);
        let job_tx = job_tx.clone();
        let sp: Vec<PathBuf> = shard_paths[rs..re].to_vec();
        let my_tensors: Vec<_> = global_tensors.iter().filter(|(_, si, _, _, _, _, _, _)| *si >= rs && *si < re).cloned().collect();
        
        Some(thread::spawn(move || {
            let mut by_shard: HashMap<usize, Vec<_>> = HashMap::new();
            for t in my_tensors { by_shard.entry(t.1).or_default().push(t); }
            for (si, tensors) in by_shard.iter() {
                let local_idx = si - rs;
                let shard_path = &sp[local_idx];
                let data = match fs::read(shard_path) { Ok(d) => d, Err(_) => continue };
                for (gi, _, name, dtype, shape, doff, bsize, dstart) in tensors {
                    let ne = (*bsize / dtype.bytes_per_element()) as usize;
                    if ne == 0 { continue; }
                    let raw = &data[(dstart + *doff as usize)..(dstart + *doff as usize + *bsize as usize)];
                    let weights = dtype.dequantize_to_f32(raw, ne);
                    let _ = job_tx.send(CompressJob { global_idx: *gi, name: name.clone(), shape: shape.clone(), element_count: ne, weights });
                }
            }
        }))
    }).collect();
    drop(job_tx);
 let weights_path = out_dir.join("weights.bin");
    let sandbag_path = out_dir.join("sandbag.bin");
    let manifest_path = out_dir.join("manifest.json");

    let mut weights_file = BufWriter::new(File::create(&weights_path)?);
    let mut sandbag_file = BufWriter::new(File::create(&sandbag_path)?);
    let mut all_stats: Vec<TensorStats> = Vec::new();
    let mut woff: u64 = 0;
    let mut moff: u64 = 0;
    let mut t_orig: u64 = 0;
    let mut t_core: u64 = 0;
    let mut t_meta: u64 = 0;
    let mut expected = 0usize;
    let mut compress_buffer: HashMap<usize, CompressResult> = HashMap::new();

    while expected < total_tensors {
        let result = match result_rx.recv() { 
            Ok(r) => r, 
            Err(_) => return Err("worker channel severed".into()) 
        };
        compress_buffer.insert(result.global_idx, result);
        
        while let Some(mut r) = compress_buffer.remove(&expected) {
            let cl = r.core.len() as u64;
            let ml = r.sandbag.len() as u64;
            weights_file.write_all(&r.core)?;
            sandbag_file.write_all(&r.sandbag)?;
            r.stats.weight_offset = woff;
            r.stats.sandbag_offset = moff;
            t_orig += r.stats.original_bytes as u64;
            t_core += cl;
            t_meta += ml;
            all_stats.push(r.stats);
            woff += cl;
            moff += ml;
            expected += 1;
        }
    }

    for h in reader_handles { let _ = h.join(); }
    for h in worker_handles { let _ = h.join(); }
    drop(weights_file);
    drop(sandbag_file);

    let overall = if t_core > 0 { t_orig as f32 / t_core as f32 } else { 1.0 };
    let final_model_name = shard_paths[0].file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
    let stats = ConversionStats { 
        model_name: final_model_name, 
        tensor_count: all_stats.len(), 
        total_original_bytes: t_orig, 
        total_core_bytes: t_core, 
        total_sandbag_bytes: t_meta, 
        overall_core_ratio: overall, 
        tensors: all_stats 
    };
    fs::write(&manifest_path, serde_json::to_string_pretty(&stats)?)?;
    
    let bin_path = out_dir.join("manifest.bin");
    if let Ok(bin_data) = bincode::serialize(&stats) {
        let _ = fs::write(bin_path, bin_data);
    }
    
    Ok(stats)
}
// Structure definition required for the quantization worker pipeline channel
struct QuantResult {
    tidx: usize,
    name: String,
    shape: Vec<usize>,
    n_elems: usize,
    dtype: u32,
    data: Vec<u8>,
    is_4bit: bool,
    group_size: usize,
}

pub fn quantize_gguf_slice_endpoint(
    gguf_path: &Path,
    out_dir: &Path,
    model_name: String,
    total_tensors: usize,
    tensor_indices: &[usize],
    result_rx: mpsc::Receiver<QuantResult>,
    handles: Vec<thread::JoinHandle<()>>,
    gguf: GGUFFile,
) -> Result<ConversionStats, Box<dyn std::error::Error>> {
    let quant_path = out_dir.join("quantized.bin");
    let manifest_path = out_dir.join("manifest.json");
    let mut quant_file = BufWriter::new(File::create(&quant_path)?);

    let mut quant_buffer: HashMap<usize, QuantResult> = HashMap::new();
    let mut all_stats: Vec<TensorStats> = Vec::with_capacity(total_tensors);
    let mut expected_idx = 0usize;
    let mut running_quant_offset = 0u64;
    let mut t_orig = 0u64;
    let mut t_quant = 0u64;

    while expected_idx < total_tensors {
        let result = match result_rx.recv() {
            Ok(r) => r,
            Err(_) => return Err("Quantization worker pool disconnected unexpectedly".into()),
        };

        let plan_position = tensor_indices.iter().position(|&idx| idx == result.tidx).unwrap();
        quant_buffer.insert(plan_position, result);

        while let Some(res) = quant_buffer.remove(&expected_idx) {
            let data_len = res.data.len();
            let orig_bytes = (res.n_elems * 4) as u64;
            quant_file.write_all(&res.data)?;

            all_stats.push(TensorStats {
                name: res.name, shape: res.shape, gguf_dtype: res.dtype, element_count: res.n_elems,
                original_bytes: orig_bytes as usize, core_bytes: data_len, sandbag_bytes: 0,
                core_ratio: orig_bytes as f32 / data_len as f32, prefix_count: 0, unique_tail_count: 0,
                shared_weights: 0, mean_precision_lost: 0.0, weight_offset: 0, sandbag_offset: 0,
                full_precision: !res.is_4bit, quant_offset: running_quant_offset, quant_bytes: data_len,
                is_4bit: res.is_4bit, group_size: res.group_size,
            });

            running_quant_offset += data_len as u64;
            t_orig += orig_bytes;
            t_quant += data_len as u64;
            expected_idx += 1;
        }
    }

    quant_file.flush()?;
    drop(quant_file);
    for h in handles { let _ = h.join(); }

    let overall_ratio = if t_quant > 0 { t_orig as f32 / t_quant as f32 } else { 1.0 };
    let conv_stats = ConversionStats { 
        model_name, 
        tensor_count: all_stats.len(), 
        total_original_bytes: t_orig, 
        total_core_bytes: t_quant, 
        total_sandbag_bytes: 0, 
        overall_core_ratio: overall_ratio, 
        tensors: all_stats 
    };
    
    fs::write(&manifest_path, serde_json::to_string_pretty(&conv_stats)?)?;
    
    let bin_path = out_dir.join("manifest.bin");
    if let Ok(bin_data) = bincode::serialize(&conv_stats) {
        let _ = fs::write(bin_path, bin_data);
    }

    let config = crate::inference::config::ModelConfig::from_gguf(&gguf);
    config.to_file(out_dir)?;
    crate::inference::tokenizer::Tokenizer::extract_to_file(&gguf, out_dir)?;
    Ok(conv_stats)
}
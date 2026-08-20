// Safetensors → compressed format conversion pipeline.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use super::common::{
    CompressJob, CompressResult, ConversionStats, TensorStats,
    compress_weights, resolve_params,
};
use super::safetensors::{SafetensorsHeader, SafetensorsDtype};

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

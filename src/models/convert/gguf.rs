// GGUF → compressed format conversion pipeline.
//
// Reader streams tensors from disk in chunks, shared work queue fans to
// N workers (GPU loader/operator + CPU workers), writer assembles in order.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use super::common::{
    CompressJob, CompressResult, CHUNK_SIZE,
    process_job,
};
use super::dedup_count::DedupCountTensor;
use super::gguf::GGUFFile;

/// Convert a full GGUF model to compressed format.
pub fn convert_gguf(
    gguf_path: &Path,
    out_dir: &Path,
    prefix_digits: usize,
    truncate_rounds: usize,
    num_workers: usize,
) -> Result<super::common::ConversionStats, Box<dyn std::error::Error>> {
    use super::common::{ConversionStats, TensorStats};

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
    let mut received = 0usize;

    let mut all_stats: Vec<TensorStats> = Vec::new();
    let mut woff: u64 = 0;
    let mut moff: u64 = 0;
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
        received += 1;
        buffer.insert(result.global_idx, (result.core, result.sandbag, result.stats));

        while let Some((core, sandbag, stats)) = buffer.remove(&expected_chunk) {
            weights_file.write_all(&core)?;
            sandbag_file.write_all(&sandbag)?;
            let (tpidx, _c) = chunk_lookup[expected_chunk];
            tensor_chunk_stats[tpidx].push(stats);
            woff += core.len() as u64;
            moff += sandbag.len() as u64;
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

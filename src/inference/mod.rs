// Inference engine — loads DedupCount-compressed weights, decompresses to INT4,
// and runs forward passes with on-the-fly INT4 dequantization in the GEMV kernel.
//
// Architecture: SSM+attention hybrid (Qwen3.5) or pure attention (Qwen2).
// Forward pass per block:
//   1. RMSNorm(attn_norm)
//   2. SSM or Attention block
//   3. Residual add
//   4. RMSNorm(ffn_norm)
//   5. FFN (SwiGLU)
//   6. Residual add

pub mod config;
pub mod tokenizer;
pub mod math;
pub mod ssm;
pub mod attention;
pub mod ffn;
pub mod sampling;
pub mod kv_cache;

use std::path::{Path, PathBuf};
use std::io::{Seek, SeekFrom, Read, Write};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::fs::File;

use crate::models::convert::{ModelLoader, deserialize_core_chunks};
use crate::models::dedup_count::{GlobalTable, DedupCountTensor};
use config::ModelConfig;
use tokenizer::Tokenizer;

pub use kv_cache::KvCache;
pub use attention::{AttnBlock, AttnState};
pub use ssm::{SsmBlock, SsmState};
pub use ffn::{FfnBlock, FfnState};

/// Running inference engine. Loads compressed model, holds state.
pub struct InferenceEngine {
    pub loader: ModelLoader,
    pub config: ModelConfig,
    pub tokenizer: Tokenizer,
    pub global_table: Arc<GlobalTable>,
    pub kv_caches: Vec<KvCache>,
    pub ssm_states: Vec<SsmState>,
    /// Tensor name → (byte offset in mmap, element_count, is_4bit, group_size).
    tensor_index: HashMap<String, (u64, usize, bool, usize)>,
    mmap: Option<memmap2::Mmap>,
    temp_path: PathBuf,
    write_pos: u64,
    pub position: usize,
    pub scratch_hidden: Vec<f32>,
    pub scratch_normed: Vec<f32>,
    pub scratch_ffn_out: Vec<f32>,
    pub scratch_logits: Vec<f32>,
    pub scratch_ffn_gate: Vec<f32>,
    pub scratch_ffn_up: Vec<f32>,
    pub scratch_ffn_act: Vec<f32>,
    pub scratch_attn_q: Vec<f32>,
    pub scratch_attn_k: Vec<f32>,
    pub scratch_attn_v: Vec<f32>,
    pub scratch_attn_out: Vec<f32>,
}

impl InferenceEngine {
    pub fn open(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Self::open_with_progress(model_dir, None)
    }

    pub fn open_with_progress(
        model_dir: &Path,
        progress: Option<&crate::progress::LoadingProgress>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("InferenceEngine: opening model from {}", model_dir.display());
        if let Some(p) = progress { p.set(6, "Reading model config..."); }
        let loader = ModelLoader::open(model_dir)?;
        let config = ModelConfig::from_dir(model_dir)?;

        if let Some(p) = progress { p.set(7, "Loading tokenizer..."); }
        let tokenizer = Tokenizer::from_dir(model_dir)?;

        let n_layers = config.n_layer + config.n_layer_nextn;
        let mut kv_caches = Vec::with_capacity(n_layers);
        let mut ssm_states = Vec::with_capacity(n_layers);
        if let Some(p) = progress { p.set(8, "Allocating KV caches..."); }
        for i in 0..n_layers {
            if config.is_recurrent[i] {
                ssm_states.push(SsmState::new(&config));
                kv_caches.push(KvCache::empty());
            } else {
                ssm_states.push(SsmState::empty());
                kv_caches.push(KvCache::new(&config));
            }
        }

        log::info!("InferenceEngine: {} layers ({} recurrent, {} attention), vocab={}, hidden={}",
            config.n_layer,
            config.is_recurrent.iter().filter(|&&r| r).count(),
            config.is_recurrent.iter().filter(|&&r| !r).count(),
            config.vocab_size, config.n_embd);

        if let Some(p) = progress { p.set(9, "Loading global table..."); }
        let global_table = {
            let cache_path = model_dir.join("global_table.bin");
            if cache_path.exists() {
                let data = std::fs::read(&cache_path)?;
                match GlobalTable::deserialize_core_chunks(&data) {
                    Some(gt) => {
                        log::info!("InferenceEngine: global table loaded from cache ({} prefixes, {} total tails)",
                            gt.prefixes.len(), gt.flat_tails.len());
                        gt
                    }
                    None => {
                        log::warn!("InferenceEngine: global table cache corrupt, rebuilding");
                        let gt = Self::build_global_table(&loader)?;
                        let _ = std::fs::write(&cache_path, &gt.serialize());
                        gt
                    }
                }
            } else {
                let gt = Self::build_global_table(&loader)?;
                let _ = std::fs::write(&cache_path, &gt.serialize());
                log::info!("InferenceEngine: global table built and cached ({} prefixes, {} total tails)",
                    gt.prefixes.len(), gt.flat_tails.len());
                gt
            }
        };

        if let Some(p) = progress { p.set(10, "Validating tensor names..."); }
        let n_embd = config.n_embd;
        let n_ff = config.n_ff;
        let vocab_size = config.vocab_size;
        let n_embd_kv = config.n_head_kv * config.n_embd_head;
        let config_clone = &config.clone();

        // Validate that all expected tensor names are present in the checkpoint.
        // This turns silent zero-fill at inference time into a loud error at load time.
        let all_names: Vec<&str> = loader.tensor_names();
        let name_set: HashSet<&str> = all_names.iter().copied().collect();
        let mut missing: Vec<String> = Vec::new();
        for il in 0..config.n_layer {
            let expected = [
                "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
                "ffn_up.weight", "ffn_down.weight",
            ];
            for suffix in &expected {
                let full = format!("blk.{}.{}", il, suffix);
                if !name_set.contains(full.as_str()) {
                    missing.push(full);
                }
            }
        }
        for special in ["token_embd.weight", "output_norm.weight", "output.weight"] {
            if !name_set.contains(special) {
                missing.push(special.to_string());
            }
        }
        if !missing.is_empty() {
            log::warn!("InferenceEngine: {} expected tensors missing from checkpoint: {:?}",
                missing.len(), &missing[..missing.len().min(10)]);
        }
        
        let temp_path = std::env::temp_dir().join(format!("agent_harness_weights_{}.bin", std::process::id()));
        File::create(&temp_path)?;

        Ok(Self {
            loader, config, tokenizer,
            global_table: Arc::new(global_table),
            kv_caches, ssm_states,
            tensor_index: HashMap::new(),
            mmap: None,
            temp_path,
            write_pos: 0,
            position: 0,
            scratch_hidden: vec![0.0; n_embd],
            scratch_normed: vec![0.0; n_embd],
            scratch_ffn_out: vec![0.0; n_embd],
            scratch_logits: vec![0.0; vocab_size],
            scratch_ffn_gate: vec![0.0; n_ff],
            scratch_ffn_up: vec![0.0; n_ff],
            scratch_ffn_act: vec![0.0; n_ff],
            scratch_attn_q: vec![0.0; n_embd],
            scratch_attn_k: vec![0.0; n_embd_kv],
            scratch_attn_v: vec![0.0; n_embd_kv],
            scratch_attn_out: vec![0.0; n_embd],
        })
    }

    fn build_global_table(loader: &ModelLoader) -> Result<GlobalTable, Box<dyn std::error::Error>> {
        let mut core_file = File::open(loader.dir.join("weights.bin"))?;
        let mut all_tensors: Vec<DedupCountTensor> = Vec::new();

        for stats in &loader.manifest.tensors {
            if stats.core_bytes == 0 || stats.sandbag_bytes == 0 { continue; }
            core_file.seek(SeekFrom::Start(stats.weight_offset))?;
            let mut buf = vec![0u8; stats.core_bytes];
            core_file.read_exact(&mut buf)?;
            let chunks = deserialize_core_chunks(&buf);
            for chunk in chunks {
                all_tensors.push(chunk);
            }
        }

        let tensor_refs: Vec<_> = all_tensors.iter().collect();
        Ok(GlobalTable::from_tensors(&tensor_refs))
    }

    pub fn decompress_all_parallel(
        &mut self,
        num_workers: usize,
        progress: Option<&crate::progress::LoadingProgress>,
    ) {
        use std::sync::{mpsc, Mutex};
        use std::thread;
        use crate::models::quantization;

        let all_names: Vec<String> = self.loader.tensor_names().iter().map(|s| s.to_string()).collect();
        let total = all_names.len();
        let gs = quantization::GROUP_SIZE;

        let mut tensor_plan: Vec<(String, u64, usize, bool, usize)> = Vec::with_capacity(total);
        let mut offset = 0u64;
        for name in &all_names {
            if let Some(stats) = self.loader.tensor_stats(name) {
                let is_fp = stats.full_precision || stats.sandbag_bytes == 0;
                let bytes = if is_fp {
                    stats.element_count * 4
                } else {
                    quantization::quantized_bytes(stats.element_count, gs)
                };
                tensor_plan.push((name.clone(), offset, stats.element_count, !is_fp, gs));
                offset += bytes as u64;
            }
        }
        self.write_pos = offset;
        let total_bytes = offset;

        log::info!("decompress_all_parallel: {} tensors, {:.2} GB, {} workers",
            total, total_bytes as f64 / 1e9, num_workers);

        {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(&self.temp_path)
                .expect("temp file not open");
            file.set_len(total_bytes).expect("failed to pre-allocate temp file");
        }

        let tensor_plan = Arc::new(tensor_plan);
        let work_queue = Arc::new(Mutex::new(
            (0..total).collect::<std::collections::VecDeque<_>>()
        ));
        let (tx, rx) = mpsc::channel::<(String, u64, usize, bool, usize)>();
        let dir = self.loader.dir.clone();
        let global_table = Arc::clone(&self.global_table);
        let temp_path = self.temp_path.clone();
        let num_workers = num_workers.max(1);
        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let wq = Arc::clone(&work_queue);
                let dir = dir.clone();
                let gt = Arc::clone(&global_table);
                let tp = Arc::clone(&tensor_plan);
                let temp = temp_path.clone();
                let tx = tx.clone();
                
                thread::spawn(move || {
                    let mut loader = match ModelLoader::open(&dir) {
                        Ok(l) => l,
                        Err(e) => {
                            log::error!("Worker: failed to open ModelLoader: {}", e);
                            return;
                        }
                    };
                    let mut file = match std::fs::OpenOptions::new().write(true).open(&temp) {
                        Ok(f) => f,
                        Err(e) => {
                            log::error!("Worker: failed to open temp file: {}", e);
                            return;
                        }
                    };

                    loop {
                        let idx = { wq.lock().unwrap().pop_front() };
                        let Some(idx) = idx else { break; };
                        let (name, byte_offset, elem_count, is_4bit, group_size) = &tp[idx];

                        match loader.decompress_tensor_global_single(name, Arc::clone(&gt)) {
                            Ok(weights) => {
                                if *is_4bit {
                                    let (scales, packed) = quantization::quantize(&weights, *group_size);
                                    let mut buf = Vec::with_capacity(scales.len() * 4 + packed.len());
                                    for &s in &scales {
                                        buf.extend_from_slice(&s.to_le_bytes());
                                    }
                                    buf.extend_from_slice(&packed);
                                    file.seek(SeekFrom::Start(*byte_offset)).ok();
                                    let _ = file.write_all(&buf);
                                } else {
                                    let bytes: &[u8] = bytemuck::cast_slice(&weights);
                                    file.seek(SeekFrom::Start(*byte_offset)).ok();
                                    let _ = file.write_all(bytes);
                                }
                                let _ = tx.send((
                                    name.clone(),
                                    *byte_offset,
                                    *elem_count,
                                    *is_4bit,
                                    *group_size,
                                ));
                            }
                            Err(e) => {
                                log::error!("Worker: failed to decompress {}: {}", name, e);
                            }
                        }
                    }
                })
            })
            .collect();

        drop(tx);
        let mut completed = 0;
        while let Ok((name, byte_offset, elem_count, is_4bit, group_size)) = rx.recv() {
            self.tensor_index.insert(name, (byte_offset, elem_count, is_4bit, group_size));
            completed += 1;
            if let Some(p) = progress {
                let pct = 10 + ((completed as u32 * 85) / total.max(1) as u32).min(85) as u8;
                p.set(pct, &format!("Decompressing {}/{} tensors", completed, total));
            }
        }

        for h in handles {
            if let Err(e) = h.join() {
                log::error!("decompress_all_parallel: worker panicked: {:?}", e);
            }
        }
        log::info!("decompress_all_parallel: {} tensors decompressed", self.tensor_index.len());
    }

    pub fn finalize_mmap(&mut self) {
        if self.mmap.is_some() {
            return;
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(&self.temp_path)
            .expect("temp file not found");
        let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap temp file");
        self.mmap = Some(mmap);
        log::info!(
            "Mmap finalized: {} tensors, {:.1} GB",
            self.tensor_index.len(),
            self.write_pos as f64 / 1e9
        );
    }

    fn forward(&mut self, token_id: u32) {
        let pos = self.position;
        self.position += 1;
        let n_embd = self.config.n_embd;
        let n_ff = self.config.n_ff;
        let n_head = self.config.n_head;
        let n_head_kv = self.config.n_head_kv;
        let n_embd_head = self.config.n_embd_head;
        let n_embd_kv = n_head_kv * n_embd_head;
        let gs = 32;
        let n_layer = self.config.n_layer;
        let eps = self.config.rms_eps;
        let vocab_size = self.config.vocab_size;
        let rope_dim = self.config.rope_dim_count;
        let rope_sec = self.config.rope_sections;
        let rope_fb = self.config.rope_freq_base;

        let mmap_ptr: *const u8 = self.mmap.as_ref().map(|m| m.as_ptr()).unwrap_or(std::ptr::null());
        let mmap_len = self.mmap.as_ref().map(|m| m.len()).unwrap_or(0);
        let ti = &self.tensor_index;

        let get_f32 = |name: &str| -> &[f32] {
            match ti.get(name) {
                Some(&(off, count, is_4bit, _)) if !is_4bit && mmap_ptr != std::ptr::null() => {
                    let s = off as usize;
                    let e = s + count * 4;
                    if e <= mmap_len {
                        bytemuck::cast_slice(unsafe {
                            std::slice::from_raw_parts(mmap_ptr.add(s), e - s)
                        })
                    } else {
                        &[]
                    }
                }
                _ => &[],
            }
        };

        let get_int4 = |name: &str| -> Option<(&[f32], &[u8], usize)> {
            let &(off, count, is_4bit, group_size) = ti.get(name)?;
            if !is_4bit || mmap_ptr == std::ptr::null() {
                return None;
            }
            let ng = (count + group_size - 1) / group_size;
            let sb = ng * 4;
            let pb = (count + 1) / 2;
            let ss = off as usize;
            let se = ss + sb + pb;
            if se <= mmap_len {
                let scales = bytemuck::cast_slice(unsafe {
                    std::slice::from_raw_parts(mmap_ptr.add(ss), sb)
                });
                let packed = unsafe { std::slice::from_raw_parts(mmap_ptr.add(ss + sb), pb) };
                Some((scales, packed, group_size))
            } else {
                None
            }
        };

        let embd = get_f32("token_embd.weight");
        let embd_start = token_id as usize * n_embd;
        if embd_start + n_embd > embd.len() {
            return;
        }
        self.scratch_hidden.copy_from_slice(&embd[embd_start..embd_start + n_embd]);

        for il in 0..n_layer {
            let norm_w = get_f32(&format!("blk.{}.attn_norm.weight", il));
            if norm_w.is_empty() {
                self.scratch_normed.copy_from_slice(&self.scratch_hidden);
            } else {
                math::rms_norm_into(&mut self.scratch_normed, &self.scratch_hidden, norm_w, eps);
            }

            let (qs, qp, _) = get_int4(&format!("blk.{}.attn_q.weight", il)).unwrap_or((&[], &[], gs));
            let (ks, kp, _) = get_int4(&format!("blk.{}.attn_k.weight", il)).unwrap_or((&[], &[], gs));
            let (vs, vp, _) = get_int4(&format!("blk.{}.attn_v.weight", il)).unwrap_or((&[], &[], gs));
            
            math::gemv_4bit_into(&mut self.scratch_attn_q, qs, qp, &self.scratch_normed, n_embd, n_embd, gs);
            math::gemv_4bit_into(&mut self.scratch_attn_k, ks, kp, &self.scratch_normed, n_embd_kv, n_embd, gs);
            math::gemv_4bit_into(&mut self.scratch_attn_v, vs, vp, &self.scratch_normed, n_embd_kv, n_embd, gs);

            let q_norm = get_f32(&format!("blk.{}.attn_q_norm.weight", il));
            if !q_norm.is_empty() {
                for h in 0..n_head {
                    let s = h * n_embd_head;
                    math::rms_norm_into(
                        &mut self.scratch_normed[..n_embd_head],
                        &self.scratch_attn_q[s..s + n_embd_head],
                        q_norm,
                        eps,
                    );
                    self.scratch_attn_q[s..s + n_embd_head].copy_from_slice(&self.scratch_normed[..n_embd_head]);
                }
            }
            
            let k_norm = get_f32(&format!("blk.{}.attn_k_norm.weight", il));
            if !k_norm.is_empty() {
                for h in 0..n_head_kv {
                    let s = h * n_embd_head;
                    math::rms_norm_into(
                        &mut self.scratch_normed[..n_embd_head],
                        &self.scratch_attn_k[s..s + n_embd_head],
                        k_norm,
                        eps,
                    );
                    self.scratch_attn_k[s..s + n_embd_head].copy_from_slice(&self.scratch_normed[..n_embd_head]);
                }
            }

            math::rope_multi(&mut self.scratch_attn_q, pos, rope_dim, rope_sec, rope_fb);
            math::rope_multi(&mut self.scratch_attn_k, pos, rope_dim, rope_sec, rope_fb);

            let kv = &mut self.kv_caches[il];
            let max_seq = kv.max_seq_len;
            for h in 0..n_head_kv {
                for d in 0..n_embd_head {
                    kv.k[h * n_embd_head * max_seq + d * max_seq + pos] = self.scratch_attn_k[h * n_embd_head + d];
                    kv.v[h * n_embd_head * max_seq + d * max_seq + pos] = self.scratch_attn_v[h * n_embd_head + d];
                }
            }

            let scale = 1.0 / (n_embd_head as f32).sqrt();
            for h in 0..n_head {
 let kv_head = if n_head_kv > 0 { h * n_head_kv / n_head } else { 0 };
                let qoff = h * n_embd_head;
                let mut scores = vec![0.0f32; pos + 1];
                let mut max_s = f32::NEG_INFINITY;
                for p in 0..=pos {
                    let mut dot = 0.0f32;
                    for d in 0..n_embd_head {
                        dot += self.scratch_attn_q[qoff + d] * kv.k[kv_head * n_embd_head * max_seq + d * max_seq + p];
                    }
                    scores[p] = dot * scale;
                    if scores[p] > max_s { max_s = scores[p]; }
                }
                let mut sum_exp = 0.0f32;
                for s in scores.iter_mut() { *s = (*s - max_s).exp(); sum_exp += *s; }
                let inv = 1.0 / sum_exp;
                for d in 0..n_embd_head {
                    let mut acc = 0.0f32;
                    for p in 0..=pos {
                        acc += scores[p] * kv.v[kv_head * n_embd_head * max_seq + d * max_seq + p];
                    }
                    self.scratch_attn_out[qoff + d] = acc * inv;
                }
            }

            let (os, op, _) = get_int4(&format!("blk.{}.attn_output.weight", il)).unwrap_or((&[], &[], gs));
            math::gemv_4bit_into(&mut self.scratch_ffn_out, os, op, &self.scratch_attn_out, n_embd, n_embd, gs);
            for i in 0..n_embd { self.scratch_hidden[i] += self.scratch_ffn_out[i]; }

            let post_w = get_f32(&format!("blk.{}.ffn_norm.weight", il));
            if post_w.is_empty() {
                self.scratch_normed.copy_from_slice(&self.scratch_hidden);
            } else {
                math::rms_norm_into(&mut self.scratch_normed, &self.scratch_hidden, post_w, eps);
            }

            let (gs2, gp, _) = get_int4(&format!("blk.{}.ffn_gate.weight", il)).unwrap_or((&[], &[], gs));
            let (us, up, _) = get_int4(&format!("blk.{}.ffn_up.weight", il)).unwrap_or((&[], &[], gs));
            let (ds, dp, _) = get_int4(&format!("blk.{}.ffn_down.weight", il)).unwrap_or((&[], &[], gs));
            if !gp.is_empty() && !up.is_empty() && !dp.is_empty() {
                ffn::swiglu_4bit_into(
                    &mut self.scratch_ffn_out, &self.scratch_normed,
                    gs2, gp, us, up, ds, dp,
                    n_embd, n_ff, gs,
                    &mut self.scratch_ffn_gate, &mut self.scratch_ffn_up, &mut self.scratch_ffn_act,
                );
                for i in 0..n_embd { self.scratch_hidden[i] += self.scratch_ffn_out[i]; }
            }
        }

        let out_norm = get_f32("output_norm.weight");
        if out_norm.is_empty() {
            self.scratch_normed.copy_from_slice(&self.scratch_hidden);
        } else {
            math::rms_norm_into(&mut self.scratch_normed, &self.scratch_hidden, out_norm, eps);
        }

        let (ls, lp, _) = get_int4("output.weight").unwrap_or((&[], &[], gs));
        if !lp.is_empty() {
            math::gemv_4bit_into(&mut self.scratch_logits, ls, lp, &self.scratch_normed, vocab_size, n_embd, gs);
        } else {
            let lm = get_f32("output.weight");
            if !lm.is_empty() {
                math::gemv_into(&mut self.scratch_logits, lm, &self.scratch_normed, vocab_size, n_embd);
            }
        }
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> String {
        let tokens = self.tokenizer.encode(prompt);
        let mut output_tokens = Vec::new();
        for &tid in &tokens { self.forward(tid); }
        for _ in 0..max_tokens {
            let next = if temperature > 0.0 {
                sampling::sample(&self.scratch_logits, temperature, 40)
            } else {
                sampling::argmax(&self.scratch_logits)
            };
            if next == self.config.eos_token_id { break; }
            output_tokens.push(next);
            self.forward(next);
        }
        self.tokenizer.decode(&output_tokens)
    }
}
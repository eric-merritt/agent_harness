use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use uuid::Uuid;
use memmap2::Mmap;
use super::avx512_kernel;
use super::common::{ConversionStats, TensorStats};
use super::core::*;
use crate::models::dedup::types::Sandbag;
use crate::memory_controller::virtual_tensor_arena::{VirtualTensorArena, PageResidency};

pub struct ModelLoader {
    pub manifest: ConversionStats,
    pub weights_mmap: Mmap,
    pub sandbag_mmap: Mmap,
    pub tensor_index: HashMap<String, usize>,
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

        let w_file = File::open(&weights_path)?;
        let s_file = File::open(&sandbag_path)?;
        let weights_mmap = unsafe { Mmap::map(&w_file)? };
        let sandbag_mmap = unsafe { Mmap::map(&s_file)? };

        Ok(Self { manifest, weights_mmap, sandbag_mmap, tensor_index, dir: dir.to_path_buf() })
    }

    pub fn bind_to_virtual_arena(&self, arena: &mut VirtualTensorArena) -> Result<(), String> {
        let mut global_page_cursor = 0usize;
        let bytes_per_page = arena.page_size as usize;

        for tensor in &self.manifest.tensors {
            let start_byte = tensor.weight_offset as usize;
            let total_bytes = tensor.core_bytes;
            let required_pages = (total_bytes + bytes_per_page - 1) / bytes_per_page;

            let end_page_idx = global_page_cursor + required_pages;
            if end_page_idx > arena.total_pages {
                return Err(format!("Arena Page Table Overflow: {}", tensor.name));
            }

            for page_idx in global_page_cursor..end_page_idx {
                let intra_offset = (page_idx - global_page_cursor) * bytes_per_page;
                let page = &mut arena.page_table[page_idx];
                page.residency = PageResidency::CpuResident; 
                page.cpu_offset = Some(start_byte + intra_offset);
            }
            global_page_cursor = end_page_idx;
        }
        Ok(())
    }

    pub fn get_tensor_slices(&self, name: &str) -> Result<(&[u8], &[u8], TensorStats), Box<dyn std::error::Error>> {
        let idx = *self.tensor_index.get(name).ok_or_else(|| format!("tensor not found: {}", name))?;
        let stats = self.manifest.tensors[idx].clone();

        let w_start = stats.weight_offset as usize;
        let w_end = w_start + stats.core_bytes;
        let s_start = stats.sandbag_offset as usize;
        let s_end = s_start + stats.sandbag_bytes;

        Ok((&self.weights_mmap[w_start..w_end], &self.sandbag_mmap[s_start..s_end], stats))
    }

    pub fn tensor_stats(&self, name: &str) -> Option<&TensorStats> {
        self.tensor_index.get(name).and_then(|&i| self.manifest.tensors.get(i))
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.manifest.tensors.iter().map(|t| t.name.as_str()).collect()
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

struct ChunkInfo {
    core_pos: usize,
    sand_pos: usize,
    element_count: usize,
    remap: Option<crate::models::dedup::types::ChunkRemap>,
}


impl ModelLoader {

    pub fn decompress_tensor(&self, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.decompress_tensor_impl(name, None, true)
    }

    pub fn decompress_tensor_global(&self, name: &str, global: Arc<crate::models::dedup::types::GlobalTable>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.decompress_tensor_impl(name, Some(global), true)
    }

    pub fn decompress_tensor_global_single(&self, name: &str, global: Arc<crate::models::dedup::types::GlobalTable>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.decompress_tensor_impl(name, Some(global), false)
    }

        fn decompress_tensor_impl(
        &self, name: &str, global: Option<Arc<crate::models::dedup::types::GlobalTable>>, parallel_chunks: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let idx = *self.tensor_index.get(name).ok_or_else(|| format!("tensor not found: {}", name))?;
        let is_full_precision = self.manifest.tensors[idx].full_precision;
        let sandbag_bytes = self.manifest.tensors[idx].sandbag_bytes;
        let core_bytes = self.manifest.tensors[idx].core_bytes;
        let weight_offset = self.manifest.tensors[idx].weight_offset as usize;
        let element_count = self.manifest.tensors[idx].element_count;

        // 1. Optimized Full-Precision Fast Escape Path
        if is_full_precision || sandbag_bytes == 0 {
            let start = weight_offset + 4;
            let end = weight_offset + core_bytes;
            let raw_slice = &self.weights_mmap[start..end];
            let mut weights = vec![0.0f32; element_count];

            if is_x86_feature_detected!("avx512f") {
                unsafe { avx512_load_full_precision(raw_slice, &mut weights) };
            } else {
                weights.iter_mut().zip(raw_slice.chunks_exact(4)).for_each(|(w, c)| {
                    *w = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                });
            }
            return Ok(weights);
        }

        let start_decompress = std::time::Instant::now();
        let (core_buf, sandbag_buf, stats) = self.get_tensor_slices(name)?;

        let mut chunks: Vec<ChunkInfo> = Vec::new();
        let mut core_pos = 4usize;
        let mut sand_pos = 0usize;
        let n_chunks = u32::from_le_bytes(core_buf[0..4].try_into().unwrap()) as usize;

        for _ in 0..n_chunks {
            if core_pos >= core_buf.len() { break; }
            let chunk_core_pos = core_pos;
            let chunk_sand_pos = sand_pos;
            let tensor = match deserialize_core_at(core_buf, &mut core_pos) { Some(t) => t, None => break };
            let chunk_sand = match Sandbag::from_bytes(&sandbag_buf[sand_pos..]) { Some(s) => s, None => break };
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

        let core_data = Arc::new(core_buf.to_vec());
        let sand_data = Arc::new(sandbag_buf.to_vec());
        let chunks_data = Arc::new(chunks);
        let offsets_data = Arc::new(offsets);
        let mut result = vec![0.0f32; stats.element_count];
        let mut total_written = 0usize;
        let mut worker_errors = 0usize;

        let use_avx512 = is_x86_feature_detected!("avx512f");

        if parallel_chunks && chunks_data.len() > 1 {
            let work_queue = Arc::new(Mutex::new((0..chunks_data.len()).collect::<VecDeque<usize>>()));
            let (result_tx, result_rx) = mpsc::channel::<(usize, Vec<f32>)>();
            let num_workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(32);

            let handles: Vec<_> = (0..num_workers).map(|_| {
                let wq = Arc::clone(&work_queue);
                let core = Arc::clone(&core_data);
                let sand = Arc::clone(&sand_data);
                let chunks_ref = Arc::clone(&chunks_data);
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
                            if let Some(ref remap) = ci.remap { gt.decompress_with_remap(&chunk_sand, &tensor, Some(remap.clone())) }
                            else { tensor.decompress_all_global(&chunk_sand, gt) }
                        } else { tensor.decompress_all(&chunk_sand) };

                        let _ = tx.send((idx, decompressed));
                    }
                })
            }).collect();
            drop(result_tx);

            // 2. Hardware-Accelerated Parallel Stitching Block
            for (idx, data) in result_rx {
                let offset = offsets_data[idx];
                let end = offset + data.len();
                if end <= result.len() {
                    if use_avx512 {
                        unsafe { avx512_stream_stitch(&data, &mut result, offset) };
                    } else {
                        result[offset..end].copy_from_slice(&data);
                    }
                    total_written += data.len();
                } else { worker_errors += 1; }
            }
            for h in handles { if h.join().is_err() { worker_errors += 1; } }
        } else {
            // 3. Hardware-Accelerated Sequential Fallback Stitching Block
            for (idx, ci) in chunks_data.iter().enumerate() {
                let mut cp = ci.core_pos;
                let tensor = match deserialize_core_at(&core_data, &mut cp) { Some(t) => t, None => continue };
                let chunk_sand = match Sandbag::from_bytes(&sand_data[ci.sand_pos..]) { Some(s) => s, None => continue };
                let decompressed = if let Some(ref gt) = global.as_ref() {
                    if let Some(ref remap) = ci.remap { gt.decompress_with_remap(&chunk_sand, &tensor, Some(remap.clone())) }
                    else { tensor.decompress_all_global(&chunk_sand, gt) }
                } else { tensor.decompress_all(&chunk_sand) };

                let offset = offsets_data[idx];
                let end = offset + decompressed.len();
                if end <= result.len() {
                    if use_avx512 {
                        unsafe { avx512_stream_stitch(&decompressed, &mut result, offset) };
                    } else {
                        result[offset..end].copy_from_slice(&decompressed);
                    }
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

}

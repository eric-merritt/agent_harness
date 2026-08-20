// ModelLoader: on-demand decompression from compressed files.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use uuid::Uuid;

use super::common::{ConversionStats, TensorStats};
use super::core::deserialize_core_at;
use super::dedup_count::Sandbag;

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

    pub fn decompress_tensor_global(&mut self, name: &str, global: Arc<super::dedup_count::GlobalTable>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.decompress_tensor_impl(name, Some(global), true)
    }

    pub fn decompress_tensor_global_single(&mut self, name: &str, global: Arc<super::dedup_count::GlobalTable>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.decompress_tensor_impl(name, Some(global), false)
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
    remap: Option<super::dedup_count::ChunkRemap>,
}

impl ModelLoader {
    fn decompress_tensor_impl(
        &mut self, name: &str, global: Option<Arc<super::dedup_count::GlobalTable>>, parallel_chunks: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let idx = *self.tensor_index.get(name)
            .ok_or_else(|| format!("tensor not found: {}", name))?;

        let is_full_precision = self.manifest.tensors[idx].full_precision;
        let sandbag_bytes = self.manifest.tensors[idx].sandbag_bytes;
        let core_bytes = self.manifest.tensors[idx].core_bytes;
        let weight_offset = self.manifest.tensors[idx].weight_offset;
        let element_count = self.manifest.tensors[idx].element_count;

        if is_full_precision || sandbag_bytes == 0 {
            self.weights_file.seek(SeekFrom::Start(weight_offset))?;
            let mut buf = vec![0u8; core_bytes];
            self.weights_file.read_exact(&mut buf)?;
            let weights: Vec<f32> = buf[4..].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            return Ok(weights);
        }

        let start_decompress = std::time::Instant::now();

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
}

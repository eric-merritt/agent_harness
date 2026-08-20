use ash::vk;

use crate::memory_controller::cpu_mem_op::CpuMemory;
use crate::memory_controller::gpu_mem_op::GpuMemory;
use crate::memory_controller::virtual_tensor_arena::{PageResidency, VirtualTensorArena};

/// Safety margin reserved for OS and other processes (2 GB).
const RAM_SAFETY_MARGIN: u64 = 2 * 1024 * 1024 * 1024;
/// VRAM headroom — never fill VRAM past this point (20% of total).
const VRAM_HEADROOM_FRACTION: f64 = 0.2;

pub struct MemoryController {
    pub gpu: GpuMemory,
    pub cpu: CpuMemory,
    pub arena: VirtualTensorArena,
}

impl MemoryController {
    pub fn new(total_virtual_size: vk::DeviceSize, page_size: vk::DeviceSize) -> Self {
        let gpu = GpuMemory::new();
        let cpu = CpuMemory::new(total_virtual_size as usize);
        let arena = unsafe {
            VirtualTensorArena::new(
                gpu.device(),
                gpu.allocator(),
                total_virtual_size,
                page_size,
            )
        };
        Self { gpu, cpu, arena }
    }

    // ── Capacity queries ───────────────────────────────────────────────

    /// Actual available system RAM (Linux: /proc/meminfo MemAvailable).
    pub fn system_available_ram() -> u64 {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemAvailable:"))
                    .and_then(|l| l.split_whitespace().nth(1).and_then(|n| n.parse::<u64>().ok()))
            })
            .map(|kb| kb * 1024)
            .unwrap_or_else(|| {
                std::fs::read_to_string("/proc/meminfo")
                    .ok()
                    .and_then(|s| {
                        s.lines()
                            .find(|l| l.starts_with("MemTotal:"))
                            .and_then(|l| l.split_whitespace().nth(1).and_then(|n| n.parse::<u64>().ok()))
                    })
                    .map(|kb| kb * 1024)
                    .unwrap_or(16 * 1024 * 1024 * 1024)
            })
    }

    /// Physical RAM currently committed by CPU-resident pages.
    pub fn cpu_committed(&self) -> u64 {
        let ps = self.arena.page_size as u64;
        self.arena.page_table.iter()
            .filter(|p| p.residency == PageResidency::CpuResident)
            .count() as u64 * ps
    }

    /// RAM available for new CPU-resident pages.
    pub fn cpu_available(&self) -> u64 {
        let system = Self::system_available_ram();
        let committed = self.cpu_committed();
        system.saturating_sub(RAM_SAFETY_MARGIN).saturating_sub(committed)
    }

    /// VRAM currently committed by GPU-resident pages.
    pub fn vram_used(&self) -> u64 {
        let ps = self.arena.page_size as u64;
        self.arena.page_table.iter()
            .filter(|p| p.residency == PageResidency::GpuResident)
            .count() as u64 * ps
    }

    /// VRAM available for new GPU-resident pages (after headroom).
    pub fn vram_available(&self) -> u64 {
        let headroom = (self.gpu.vram_capacity() as f64 * VRAM_HEADROOM_FRACTION) as u64;
        self.gpu.vram_capacity().saturating_sub(headroom).saturating_sub(self.vram_used())
    }

    // ── Space management ───────────────────────────────────────────────

    /// Free CPU RAM by uploading CPU-resident pages to GPU.
    /// Each uploaded page's CPU copy is dropped via madvise to reclaim physical RAM.
    /// Does NOT cascade into free_vram_space — if GPU commit fails, skips the page.
    pub fn free_cpu_space(&mut self, bytes: u64) -> u64 {
        let page_size = self.arena.page_size as usize;
        let mut freed = 0u64;
        if freed >= bytes { return freed; }

        for page_index in 0..self.arena.total_pages {
            if freed >= bytes { break; }
            let residency = self.arena.page_table[page_index].residency;
            if residency != PageResidency::CpuResident { continue; }

            let data = self.cpu.read_page(page_index, page_size).to_vec();

            unsafe {
                self.arena.commit_page(self.gpu.device(), self.gpu.queue(), page_index);
            }

            let page = &self.arena.page_table[page_index];
            if page.residency != PageResidency::GpuResident {
                continue;
            }

            let offset = page_index as vk::DeviceSize * self.arena.page_size;
            unsafe {
                self.gpu.upload(self.arena.sparse_buffer, offset, &data);
            }

            self.cpu.drop_page(page_index, page_size);
            freed += page_size as u64;
        }

        if freed < bytes {
            log::warn!(
                "free_cpu_space: needed {} bytes, only freed {} — system RAM may be exhausted",
                bytes, freed
            );
        }
        freed
    }

    /// Free VRAM by evicting GPU-resident pages to CPU.
    /// Ensures CPU RAM is available for the incoming data first.
    pub fn free_vram_space(&mut self, bytes: u64, exclude_page: usize) -> u64 {
        let page_size = self.arena.page_size as usize;
        let mut freed = 0u64;

        for page_index in 0..self.arena.total_pages {
            if freed >= bytes { break; }
            if page_index == exclude_page { continue; }

            let residency = self.arena.page_table[page_index].residency;
            if residency != PageResidency::GpuResident { continue; }

            let needed = page_size as u64;
            if self.cpu_available() < needed {
                let deficit = needed - self.cpu_available();
                self.free_cpu_space(deficit);
            }

            let data = self.download_page(page_index);
            self.cpu.write_page(page_index, page_size, &data);
            self.evict_page(page_index);

            let page = &mut self.arena.page_table[page_index];
            page.residency = PageResidency::CpuResident;
            page.cpu_offset = Some(page_index * page_size);

            freed += page_size as u64;
        }

        if freed < bytes {
            log::warn!(
                "free_vram_space: needed {} bytes, only freed {} — VRAM may be exhausted",
                bytes, freed
            );
        }
        freed
    }

    // ── Page operations ────────────────────────────────────────────────

    /// Commit a page to GPU VRAM. Falls back to CPU on OOM.
    pub fn commit_page(&mut self, page_index: usize) {
        unsafe {
            self.arena.commit_page(self.gpu.device(), self.gpu.queue(), page_index);
        }
    }

    /// Evict a page from GPU — unbind and free VRAM. Data is NOT preserved.
    pub fn evict_page(&mut self, page_index: usize) {
        unsafe {
            self.arena.evict_page(
                page_index,
                self.gpu.allocator(),
                self.gpu.queue(),
                self.gpu.device(),
            );
        }
    }

    /// Upload data to a GPU page. Commits the page first if needed, then uploads.
    /// After successful upload, drops the CPU copy to free physical RAM.
    /// If VRAM is full, evicts a cold GPU page to make space.
    pub fn upload_page(&mut self, page_index: usize, data: &[u8]) {
        let page_size = self.arena.page_size as usize;

        let residency = self.arena.page_table[page_index].residency;
        if residency == PageResidency::Unmapped {
            let needed = page_size as u64;
            if self.vram_available() < needed {
                self.free_vram_space(needed, page_index);
            }
            self.commit_page(page_index);
        }

        let residency = self.arena.page_table[page_index].residency;
        if residency != PageResidency::GpuResident {
            log::warn!("upload_page: page {} not GPU-resident after commit, keeping on CPU", page_index);
            self.cpu.write_page(page_index, page_size, data);
            let page = &mut self.arena.page_table[page_index];
            page.residency = PageResidency::CpuResident;
            page.cpu_offset = Some(page_index * page_size);
            return;
        }

        let offset = page_index as vk::DeviceSize * self.arena.page_size;
        unsafe {
            self.gpu.upload(self.arena.sparse_buffer, offset, data);
        }

        self.cpu.drop_page(page_index, page_size);
    }

    /// Download a page's data from GPU. Page must be GPU-resident.
    pub fn download_page(&self, page_index: usize) -> Vec<u8> {
        let offset = page_index as vk::DeviceSize * self.arena.page_size;
        let size = self.arena.page_size;
        unsafe {
            self.gpu.download(self.arena.sparse_buffer, offset, size)
        }
    }

    /// Evict a GPU page but preserve its data in CPU memory.
    /// Ensures CPU RAM is available before downloading.
    pub fn evict_page_with_data(&mut self, page_index: usize) {
        let page_size = self.arena.page_size as usize;

        let needed = page_size as u64;
        if self.cpu_available() < needed {
            let deficit = needed - self.cpu_available();
            self.free_cpu_space(deficit);
        }

        let data = self.download_page(page_index);
        self.cpu.write_page(page_index, page_size, &data);
        self.evict_page(page_index);

        let page = &mut self.arena.page_table[page_index];
        page.residency = PageResidency::CpuResident;
        page.cpu_offset = Some(page_index * page_size);
    }

    /// Promote a CPU-resident page to GPU, freeing its CPU RAM.
    pub fn migrate_to_gpu(&mut self, page_index: usize) {
        let page_size = self.arena.page_size as usize;
        let residency = self.arena.page_table[page_index].residency;
        if residency != PageResidency::CpuResident { return; }

        let data = self.cpu.read_page(page_index, page_size).to_vec();

        let page = &mut self.arena.page_table[page_index];
        page.residency = PageResidency::Unmapped;
        page.cpu_offset = None;

        self.upload_page(page_index, &data);
    }

    /// Demote a GPU-resident page to CPU, preserving data.
    pub fn migrate_to_cpu(&mut self, page_index: usize) {
        let residency = self.arena.page_table[page_index].residency;
        if residency != PageResidency::GpuResident { return; }
        self.evict_page_with_data(page_index);
    }

    /// Read a page's data from wherever it lives. Returns owned bytes.
    pub fn read_page(&self, page_index: usize) -> Vec<u8> {
        let page_size = self.arena.page_size as usize;
        let residency = self.arena.page_table[page_index].residency;
        match residency {
            PageResidency::CpuResident => {
                self.cpu.read_page(page_index, page_size).to_vec()
            }
            PageResidency::GpuResident => {
                self.download_page(page_index)
            }
            PageResidency::Unmapped => {
                panic!("Cannot read unmapped page {}", page_index);
            }
        }
    }

    /// Write data to a page. Routes to CPU or GPU based on current residency.
    /// If unmapped, writes to CPU and marks it CPU-resident.
    pub fn write_page(&mut self, page_index: usize, data: &[u8]) {
        let page_size = self.arena.page_size as usize;
        let residency = self.arena.page_table[page_index].residency;
        match residency {
            PageResidency::GpuResident => {
                let offset = page_index as vk::DeviceSize * self.arena.page_size;
                unsafe {
                    self.gpu.upload(self.arena.sparse_buffer, offset, data);
                }
            }
            PageResidency::CpuResident | PageResidency::Unmapped => {
                self.cpu.write_page(page_index, page_size, data);
                let page = &mut self.arena.page_table[page_index];
                page.residency = PageResidency::CpuResident;
                page.cpu_offset = Some(page_index * page_size);
            }
        }
    }

    /// Place a page on GPU or CPU based on the target hint.
    /// On GPU: ensures VRAM space, commits, uploads, drops CPU copy.
    /// On CPU: ensures RAM space, writes, marks CPU-resident.
    pub fn place_page(&mut self, page_index: usize, data: &[u8], on_gpu: bool) {
        if on_gpu {
            self.upload_page(page_index, data);
        } else {
            let page_size = self.arena.page_size as usize;
            let needed = page_size as u64;
            if self.cpu_available() < needed {
                let deficit = needed - self.cpu_available();
                self.free_cpu_space(deficit);
            }
            self.cpu.write_page(page_index, page_size, data);
            let page = &mut self.arena.page_table[page_index];
            page.residency = PageResidency::CpuResident;
            page.cpu_offset = Some(page_index * page_size);
        }
    }

    /// GPU buffer handle + offset for shader binding of a GPU-resident page.
    pub fn gpu_binding(&self, page_index: usize) -> (vk::Buffer, vk::DeviceSize) {
        let offset = page_index as vk::DeviceSize * self.arena.page_size;
        (self.arena.sparse_buffer, offset)
    }
}

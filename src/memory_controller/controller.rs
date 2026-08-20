use ash::vk;
use std::sync::{Arc, Mutex};
use crate::memory_controller::virtual_tensor_arena::{VirtualTensorArena, PageResidency, OperationType};

// Mock structures to represent your internal architecture fields
pub struct GpuContext;
pub struct CpuMemoryManager;

impl GpuContext {
    pub fn device(&self) -> &ash::Device { todo!() }
    pub fn queue(&self) -> vk::Queue { todo!() }
    pub fn allocator(&self) -> Arc<Mutex<gpu_allocator::vulkan::Allocator>> { todo!() }
    pub unsafe fn upload(&self, _buf: vk::Buffer, _offset: vk::DeviceSize, _data: &[u8]) { todo!() }
    pub unsafe fn download(&self, _buf: vk::Buffer, _offset: vk::DeviceSize, _size: vk::DeviceSize) -> Vec<u8> { todo!() }
}

impl CpuMemoryManager {
    pub fn write_page(&mut self, _idx: usize, _size: usize, _data: &[u8]) { todo!() }
    pub fn read_page(&self, _idx: usize, _size: usize) -> &[u8] { todo!() }
    pub fn drop_page(&mut self, _idx: usize, _size: usize) { todo!() }
}

pub struct MemoryController {
    pub arena: VirtualTensorArena,
    pub gpu: GpuContext,
    pub cpu: CpuMemoryManager,
    pub max_cpu_bytes: u64,
    pub used_cpu_bytes: u64,
    pub max_vram_bytes: u64,
    pub used_vram_bytes: u64,
}

impl MemoryController {
    pub fn cpu_available(&self) -> u64 {
        self.max_cpu_bytes.saturating_sub(self.used_cpu_bytes)
    }

    pub fn vram_available(&self) -> u64 {
        self.max_vram_bytes.saturating_sub(self.used_vram_bytes)
    }

    pub fn free_cpu_space(&mut self, _bytes: u64) {
        // Implementation for cleaning or paging out cold RAM blocks
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
        let op_type = OperationType::Drop;
        
        unsafe {
            self.arena.evict_page(
                page_index,
                self.gpu.allocator(),
                self.gpu.queue(),
                self.gpu.device(),
                op_type
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

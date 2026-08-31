use memmap2::{MmapMut, MmapOptions};
use sysinfo::System;

// Safely match libc signatures across target platforms
#[cfg(target_family = "unix")]
unsafe extern "C" {
	fn madvise(addr: *mut std::ffi::c_void, length: usize, advice: i32) -> i32;
}

#[cfg(target_family = "unix")]
const MADV_DONTNEED: i32 = 4;

/// CPU backing pool for the VirtualTensorArena.
pub struct CpuMemory {
	pool: MmapMut,
	total_size: usize,
}

impl CpuMemory {
	/// Instantiates a raw anonymous virtual memory reservation of size `total_size`
	pub fn new(total_size: usize) -> Result<Self, std::io::Error> {
		if total_size == 0 {
			return Err(std::io::Error::new(
				std::io::ErrorKind::InvalidInput,
				"Cannot allocate a 0-byte memory arena",
			));
		}

		// Allocate a completely anonymous, unmapped backing zone in virtual memory
		let pool = 
			MmapOptions::new()
				.len(total_size)
				.map_anon()?;

		Ok(Self {
			pool,
			total_size,
		})
	}

	/// Query the underlying Operating System for immediate available bytes
	pub fn get_avail_cpu_mem() -> usize {
		// Use a local builder to avoid refreshing entire process trees unnecessarily
		let mut system = System::new();
		system.refresh_memory(); // Only fetch memory statistics for maximum speed
		system.available_memory() as usize
	}	
	
	pub fn capacity(&self) -> usize {
		self.total_size
	}

	/// Read-only slice at `offset` for `len` bytes.
	pub fn slice_at(&self, offset: usize, len: usize) -> &[u8] {
		&self.pool[offset..offset + len]
	}

	/// Mutable slice at `offset` for `len` bytes.
	pub fn mut_slice_at(&mut self, offset: usize, len: usize) -> &mut [u8] {
		&mut self.pool[offset..offset + len]
	}

	/// Write data into the pool at `offset`.
	pub fn write_at(&mut self, offset: usize, data: &[u8]) {
		self.pool[offset..offset + data.len()].copy_from_slice(data);
	}

	/// Read a page by its index. The page_size must match the VTA's.
	pub fn read_page(&self, page_index: usize, page_size: usize) -> &[u8] {
		let offset = page_index * page_size;
		&self.pool[offset..offset + page_size]
	}

	/// Write a page by its index. Copies `data` into the first `data.len()` bytes
	/// of the page (padded to `page_size`). Caller must ensure `data.len() <= page_size`.
	pub fn write_page(&mut self, page_index: usize, page_size: usize, data: &[u8]) {
		let offset = page_index * page_size;
		let end = (offset + data.len()).min(self.total_size);
		self.pool[offset..end].copy_from_slice(&data[..end - offset]);
	}

	/// Raw read pointer at `offset`.
	pub fn ptr_at(&self, offset: usize) -> *const u8 {
		unsafe { self.pool.as_ptr().add(offset) }
	}

	/// Raw write pointer at `offset`.
	pub fn mut_ptr_at(&mut self, offset: usize) -> *mut u8 {
		unsafe { self.pool.as_mut_ptr().add(offset) }
	}

	/// Drop the physical RAM backing for a page, freeing system memory.
	#[cfg(target_family = "unix")]
	pub fn drop_page(&mut self, page_index: usize, page_size: usize) {
		let offset = page_index * page_size;
		let ptr = unsafe { self.pool.as_mut_ptr().add(offset) };
		let ret = unsafe { madvise(ptr as *mut std::ffi::c_void, page_size, MADV_DONTNEED) };
		if ret != 0 {
			log::warn!(
				"madvise(MADV_DONTNEED) failed for page {}: {}",
				page_index,
				ret
			);
		}
	}

	// Fallback platform compilation safety flag loop
	#[cfg(not(target_family = "unix"))]
	pub fn drop_page(&mut self, _page_index: usize, _page_size: usize) {
		// Non-linux/unix systems skip madvise backing cleanup natively
	}
}

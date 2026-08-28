use memmap2::MmapMut;

unsafe extern "C" {
	fn madvise(addr: *mut std::ffi::c_void, length: usize, advice: i32) -> i32;
}

const MADV_DONTNEED: i32 = 4;

/// CPU backing pool for the VirtualTensorArena.
///
/// A single anonymous mmap reservation that mirrors the VTA's virtual address space.
/// When pages are routed to CPU (GPU OOM or eviction), their data lives here at
/// `page_index * page_size`. The VTA tracks which pages are CPU-resident; this
/// struct just provides raw read/write access to the underlying bytes.
///
/// `drop_page` uses madvise(MADV_DONTNEED) to release physical RAM backing a page
/// while keeping the virtual address valid. Reads of a dropped page return zeros.
/// This lets us free CPU RAM when a page has been uploaded to GPU.
pub struct CpuMemory {
	pool: MmapMut,
	total_size: usize,
}

impl CpuMemory {
	pub fn new(total_size: usize) -> Self {
		let pool =
			MmapMut::map_anon(total_size).expect("Failed to reserve CPU virtual memory space");
		Self { pool, total_size }
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

	/// Write a page by its index.
	pub fn write_page(&mut self, page_index: usize, page_size: usize, data: &[u8]) {
		let offset = page_index * page_size;
		self.pool[offset..offset + data.len()].copy_from_slice(data);
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
	/// The virtual address space remains valid but reads return zeros.
	/// Call this after uploading a page to GPU to reclaim CPU RAM.
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
}

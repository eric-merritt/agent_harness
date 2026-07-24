// Imports needed: zerocopy traits for zero-copy operations, bytemuck for safe casting, MemoryDevice from device module
// This module defines the central MemoryController for unified GPU+CPU memory management

/// Struct managing unified memory pool across GPU and CPU devices
/// Used by all modules requiring high-performance memory access
pub struct MemoryController {
    // Map of device ID to MemoryDevice instance
    devices: dashmap::DashMap<String, MemoryDevice>,
    // Primary unified memory allocator
    allocator: MemoryAllocator,
    // Total managed memory in bytes
    total_capacity: usize,
    // Currently allocated memory in bytes
    allocated_bytes: usize,
    // Flag indicating if unified memory mode is active
    unified_mode: bool,
}

/// Implements MemoryController with allocation and device management
impl MemoryController {
    /// Creates new MemoryController with detected devices
    /// Returns Result<MemoryController, Error> - fails if no devices found
    pub fn new() -> Result<Self, Box<dyn std::error::Error>>;

    /// Registers a new memory device (GPU or CPU)
    /// Takes &mut self and MemoryDevice, returns Result<(), Error>
    pub fn register_device(&mut self, device: MemoryDevice) -> Result<(), Box<dyn std::error::Error>>;

    /// Allocates memory from the unified pool
    /// Takes &mut self, size: usize, MemoryType, returns Result<u64, Error> with handle ID
    pub fn allocate(&mut self, size: usize, memory_type: MemoryType) -> Result<u64, Box<dyn std::error::Error>>;

    /// Deallocates memory by handle ID
    /// Takes &mut self and u64 handle_id, returns Result<(), Error>
    pub fn deallocate(&mut self, handle_id: u64) -> Result<(), Box<dyn std::error::Error>>;

    /// Gets a ZeroCopyBuffer reference for a handle
    /// Takes &self and u64 handle_id, returns Result<&ZeroCopyBuffer, Error>
    pub fn get_buffer(&self, handle_id: u64) -> Result<&ZeroCopyBuffer, Box<dyn std::error::Error>>;

    /// Gets mutable ZeroCopyBuffer reference for a handle
    /// Takes &mut self and u64 handle_id, returns Result<&mut ZeroCopyBuffer, Error>
    pub fn get_buffer_mut(&mut self, handle_id: u64) -> Result<&mut ZeroCopyBuffer, Box<dyn std::error::Error>>;

    /// Copies data between two memory handles efficiently
    /// Takes &mut self, src_id: u64, dst_id: u64, returns Result<(), Error>
    pub fn copy_between(&mut self, src_id: u64, dst_id: u64) -> Result<(), Box<dyn std::error::Error>>;

    /// Returns total allocated memory in bytes
    /// Takes &self, returns usize
    pub fn allocated_bytes(&self) -> usize;

    /// Returns total available memory in bytes
    /// Takes &self, returns usize
    pub fn available_bytes(&self) -> usize;

    /// Enables unified memory mode combining GPU+CPU pools
    /// Takes &mut self, returns Result<(), Error>
    pub fn enable_unified_mode(&mut self) -> Result<(), Box<dyn std::error::Error>>;

    /// Disables unified memory mode, separates pools
    /// Takes &mut self, returns Result<(), Error>
    pub fn disable_unified_mode(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}

/// Statistics struct for memory monitoring
/// Used by MemoryController to track allocation patterns
#[derive(Clone, Debug, Default)]
pub struct MemoryStats {
    /// Total allocations count
    pub allocation_count: u64,
    /// Total deallocations count
    pub deallocation_count: u64,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Current fragmentation ratio (0.0-1.0)
    pub fragmentation: f64,
    /// Average allocation size in bytes
    pub avg_allocation_size: f64,
}

/// Implements MemoryStats with calculation methods
impl MemoryStats {
    /// Creates empty MemoryStats
    /// Returns MemoryStats with zero values
    pub fn new() -> Self;

    /// Updates statistics after allocation
    /// Takes &mut self, size: usize, no return
    pub fn record_allocation(&mut self, size: usize);

    /// Updates statistics after deallocation
    /// Takes &mut self, size: usize, no return
    pub fn record_deallocation(&mut self, size: usize);
}

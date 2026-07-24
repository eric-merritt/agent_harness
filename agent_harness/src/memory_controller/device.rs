// Imports needed: serde for serialization, MemoryType from event_core events module
// This module defines memory device abstraction for GPU and CPU

/// Enum representing different types of memory devices
/// Used by MemoryController to route allocations to appropriate hardware
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DeviceType {
    /// System RAM - general purpose, larger capacity
    Cpu,
    /// GPU VRAM - high bandwidth, parallel access
    Gpu,
    /// Unified memory accessible by both CPU and GPU
    Unified,
    /// Specialized tensor cores for AI operations
    TensorCore,
}

/// Struct representing a physical or virtual memory device
/// Used by MemoryController to manage device-specific allocation
pub struct MemoryDevice {
    /// Unique device identifier
    pub id: String,
    /// Type of device (CPU, GPU, etc.)
    pub device_type: DeviceType,
    /// Total capacity in bytes
    pub total_bytes: usize,
    /// Available capacity in bytes
    pub available_bytes: usize,
    /// Memory bandwidth in bytes per second
    pub bandwidth_bps: u64,
    /// Latency in nanoseconds
    pub latency_ns: u32,
    /// Flag indicating if device supports zero-copy
    pub zero_copy_capable: bool,
}

/// Implements MemoryDevice with capacity tracking
impl MemoryDevice {
    /// Creates new MemoryDevice with specified parameters
    /// Takes id: String, device_type: DeviceType, total_bytes: usize, returns MemoryDevice
    pub fn new(id: String, device_type: DeviceType, total_bytes: usize) -> Self;

    /// Allocates space on this device
    /// Takes &mut self and size: usize, returns Result<(), Error> if insufficient space
    pub fn allocate(&mut self, size: usize) -> Result<(), Box<dyn std::error::Error>>;

    /// Deallocates space on this device
    /// Takes &mut self and size: usize, returns Result<(), Error>
    pub fn deallocate(&mut self, size: usize) -> Result<(), Box<dyn std::error::Error>>;

    /// Returns utilization percentage (0.0-1.0)
    /// Takes &self, returns f64
    pub fn utilization(&self) -> f64;

    /// Checks if device can accommodate requested size
    /// Takes &self and size: usize, returns bool
    pub fn can_allocate(&self, size: usize) -> bool;
}

/// Trait for device-specific memory operations
/// Implemented by GPU and CPU specific backends
pub trait DeviceBackend: Send + Sync {
    /// Returns device type identifier
    /// Takes &self, returns DeviceType
    fn device_type(&self) -> DeviceType;

    /// Performs device-specific allocation
    /// Takes &mut self and size: usize, returns Result<u64, Error> with handle
    fn allocate_raw(&mut self, size: usize) -> Result<u64, Box<dyn std::error::Error>>;

    /// Performs device-specific deallocation
    /// Takes &mut self and handle: u64, returns Result<(), Error>
    fn deallocate_raw(&mut self, handle: u64) -> Result<(), Box<dyn std::error::Error>>;

    /// Copies data to host memory
    /// Takes &self, handle: u64, returns Result<Vec<u8>, Error>
    fn read_to_host(&self, handle: u64) -> Result<Vec<u8>, Box<dyn std::error::Error>>;

    /// Copies data from host memory to device
    /// Takes &mut self, handle: u64, data: &[u8], returns Result<(), Error>
    fn write_from_host(&mut self, handle: u64, data: &[u8]) -> Result<(), Box<dyn std::error::Error>>;
}

/// CPU-specific backend implementation
pub struct CpuBackend {
    // Internal memory mapping
    mappings: std::collections::HashMap<u64, Vec<u8>>,
    // Next handle ID counter
    next_handle: u64,
}

/// GPU-specific backend implementation  
pub struct GpuBackend {
    // Placeholder for GPU buffer handles (actual impl requires cuda/wgpu)
    buffer_handles: std::collections::HashMap<u64, ()>,
    // Next handle ID counter
    next_handle: u64,
}

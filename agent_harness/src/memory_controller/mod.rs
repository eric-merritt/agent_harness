// Module declaration for memory_controller - zero-copy memory management across GPU/CPU
// This module handles unified memory abstraction and direct memory access

pub mod controller;
// Defines MemoryController struct for unified GPU+CPU RAM management

pub mod device;
// Defines MemoryDevice enum and traits for GPU/CPU device abstraction

pub mod buffer;
// Defines ZeroCopyBuffer struct with zerocopy trait implementations

pub mod allocator;
// Defines MemoryAllocator with pool-based allocation strategies

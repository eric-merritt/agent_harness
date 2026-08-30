use crate::memory_controller::virtual_tensor_arena::VirtualPage;

struct TensorBlock {
	ModelSizeBuffer: u32,
	GPUWorkPool: VirtualPage,
	
}
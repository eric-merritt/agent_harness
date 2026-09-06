pub fn conv1d_depthwise(
	input: &[f32],
	kernel: &[f32],
	kernel_size: usize,
	channels: usize,
	state: &mut Vec<f32>,
) -> Vec<f32> {
	let mut output = vec![0.0f32; channels];
	conv1d_depthwise_into(&mut output, input, kernel, kernel_size, channels, state);
	output
}

pub fn conv1d_depthwise_into(
	out: &mut [f32],
	input: &[f32],
	kernel: &[f32],
	kernel_size: usize,
	channels: usize,
	state: &mut Vec<f32>,
) {
	if kernel_size < 1 {
		return;
	}

	for c in 0..channels {
		let mut sum = 0.0f32;
		for k in 0..kernel_size - 1 {
			sum += state[k * channels + c] * kernel[c * kernel_size + k];
		}
		sum += input[c] * kernel[c * kernel_size + (kernel_size - 1)];
		out[c] = sum;
	}

	let history_steps = kernel_size - 1;
	if history_steps > 1 {
		state.copy_within(channels..history_steps * channels, 0);
	}

	if history_steps > 0 {
		let start_idx = (history_steps - 1) * channels;
		state[start_idx..history_steps * channels].copy_from_slice(input);
	}
}

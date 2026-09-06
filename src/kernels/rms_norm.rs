pub fn rms_norm_into(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
	let n = x.len();
	let mut ss = 0.0f32;
	for &v in x {
		ss += v * v;
	}
	let inv = 1.0f32 / ((ss / n as f32) + eps).sqrt();
	for i in 0..n {
		out[i] = x[i] * inv * weight[i];
	}
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
	let mut out = vec![0.0f32; x.len()];
	rms_norm_into(&mut out, x, weight, eps);
	out
}

pub fn l2_norm(x: &mut [f32], eps: f32) {
	let mut norm = 0.0f32;
	for &v in x.iter() {
		norm += v * v;
	}
	let inv = 1.0f32 / norm.sqrt().max(eps);
	for v in x.iter_mut() {
		*v *= inv;
	}
}

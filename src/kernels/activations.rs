#[inline]
pub fn silu(x: f32) -> f32 {
	x * sigmoid(x)
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
	if x >= 0.0 {
		1.0 / (1.0 + (-x).exp())
	} else {
		let e = x.exp();
		e / (1.0 + e)
	}
}

#[inline]
pub fn softplus(x: f32) -> f32 {
	if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

pub fn softmax(x: &mut [f32]) {
	let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
	let mut sum = 0.0f32;
	for v in x.iter_mut() {
		*v = (*v - max).exp();
		sum += *v;
	}
	let inv = 1.0f32 / sum;
	for v in x.iter_mut() {
		*v *= inv;
	}
}

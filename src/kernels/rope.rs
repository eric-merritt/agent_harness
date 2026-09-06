pub fn rope_multi(x: &mut [f32], pos: usize, _n_rot: usize, sections: [i32; 4], freq_base: f32) {
	let mut total_d = 0usize;
	for &section in &sections {
		if section > 0 {
			total_d += section as usize;
		}
	}
	if total_d == 0 {
		return;
	}

	let mut offset = 0usize;
	let mut freq_idx = 0usize;

	for &section in &sections {
		let d = if section <= 0 { 0 } else { section as usize };
		if d == 0 {
			continue;
		}

		for _ in 0..d {
			let dim_idx = offset + 2 * (freq_idx - offset / 2);
			if dim_idx + 1 >= x.len() {
				break;
			}

			let theta = pos as f32 * freq_base.powf(-(freq_idx as f32) / (total_d as f32));
			let (sin, cos) = theta.sin_cos();

			let x0 = x[dim_idx];
			let x1 = x[dim_idx + 1];
			x[dim_idx] = x0 * cos - x1 * sin;
			x[dim_idx + 1] = x0 * sin + x1 * cos;

			freq_idx += 1;
		}
		offset += 2 * d;
	}
}

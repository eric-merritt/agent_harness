//! CPU-side memory conversion helpers.
//!
//! Scalar reconstruction paths, f16/f32 conversion, etc.

/// Convert f32 to IEEE 754 binary16 (half precision).
/// Handles exponent bias shift (127 → 15), mantissa rounding, and special values.
#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
	let bits = v.to_bits();
	let sign = ((bits >> 16) & 0x8000) as u16;
	let exp = ((bits >> 23) & 0xFF) as i32;
	let mant = (bits & 0x7FFFFF) as u32;

	// Extract f32 exponent (unbiased)
	let e = exp as i32 - 127;

	if exp == 0 {
		// Zero or subnormal → flush to zero
		return sign;
	}
	if exp == 0xFF {
		// Inf or NaN → preserve
		return sign | (0x7C00u16 | ((mant >> 13) & 0x03FF) as u16);
	}

	// Check for overflow into f16
	if e >= 15 {
		return sign | 0x7C00; // ±infinity
	}
	if e <= -15 {
		return sign; // Flush to zero for underflow
	}

	// Re-bias exponent for f16 (bias 15)
	let new_exp = (e + 15) as u16;
	// Round mantissa: f32 has 23 mantissa bits, f16 has 10. Shift right by 13.
	// Bit 13 (the guard bit) determines rounding.
	let new_mant = ((mant >> 13) & 0x3FF) as u16;
	let guard = (mant >> 12) & 1; // Round bit
	let rounded_mant = if guard == 1 {
		// Round to nearest, ties to even
		if new_mant & 1 == 0 {
			new_mant
		} else {
			new_mant.wrapping_add(1)
		}
	} else {
		new_mant
	};

	sign | (new_exp << 10) | rounded_mant
}

/// Reconstruct f32 weights from GPU GPU prefix/tail/sign arrays (scalar path).
pub fn reconstruct_from_gpu_scalar(
	prefix_ints: &[u32],
	tails: &[u32],
	signs: &[u32],
	prefix_scale: f32,
) -> Vec<f32> {
	let n = prefix_ints.len();
	(0..n)
		.map(|i| {
			let prefix_val = (prefix_ints[i] as f32) / prefix_scale;
			let tail_val = (tails[i] as f32) / 10_000_000.0;
			let abs_w = prefix_val + tail_val;
			if signs[i] != 0 {
				-abs_w
			} else {
				abs_w
			}
		})
		.collect()
}

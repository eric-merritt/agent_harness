// Logit sampling: argmax, temperature, top-k.

/// Greedy: pick the highest logit.
pub fn argmax(logits: &[f32]) -> u32 {
	let mut best = 0u32;
	let mut best_val = f32::NEG_INFINITY;
	for (i, &v) in logits.iter().enumerate() {
		if v > best_val {
			best_val = v;
			best = i as u32;
		}
	}
	best
}

/// Temperature + top-k sampling.
pub fn sample(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
	let n = logits.len();
	if n == 0 {
		return 0;
	}

	// Find max for numerical stability before exp
	let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

	// Apply temperature + exp (shifted by max to prevent overflow)
	let inv_temp = 1.0f32 / temperature;
	let probs: Vec<f32> = logits
		.iter()
		.map(|&v| ((v - max_logit) * inv_temp).exp())
		.collect();

	// Isolate top-k candidates using select_nth_unstable
	let mut indices: Vec<usize> = (0..n).collect();
	let k = top_k.min(n);
	if k > 0 {
		indices.select_nth_unstable_by(k - 1, |&a, &b| {
			probs[b]
				.partial_cmp(&probs[a])
				.unwrap_or(std::cmp::Ordering::Equal)
		});
		indices.truncate(k);

		// CRITICAL FIX: Sort the sliced subset into descending order to ensure
		// the CDF selection iteration properly respects descending weight priorities.
		indices.sort_by(|&a, &b| {
			probs[b]
				.partial_cmp(&probs[a])
				.unwrap_or(std::cmp::Ordering::Equal)
		});
	}

	// Softmax over top-k
	let mut sum = 0.0f32;
	for &i in &indices {
		sum += probs[i];
	}
	if sum == 0.0 {
		return indices[0] as u32;
	}
	let inv = 1.0f32 / sum;

	// Sample via CDF scanning
	let r: f32 = rand_val();
	let mut cdf = 0.0f32;
	for &i in &indices {
		cdf += probs[i] * inv;
		if r < cdf {
			return i as u32;
		}
	}
	*indices.last().unwrap() as u32
}

/// Simple PRNG (xorshift) initialized dynamically with system clocks to guarantee non-determinism.
fn rand_val() -> f32 {
	use std::sync::atomic::{AtomicU64, Ordering};
	use std::time::{SystemTime, UNIX_EPOCH};

	// Lazy initialize the atomic seed state using a non-deterministic Unix timestamp baseline
	static SEED: AtomicU64 = AtomicU64::new(0);
	if SEED.load(Ordering::Relaxed) == 0 {
		let now = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.map(|d| d.as_nanos())
			.unwrap_or(1337);
		let _ = SEED.compare_exchange(0, (now as u64) | 1, Ordering::Relaxed, Ordering::Relaxed);
	}

	let mut x = SEED.load(Ordering::Relaxed);
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	SEED.store(x, Ordering::Relaxed);

	// Map safely into standard IEEE floating point range [0.0, 1.0)
	(x & 0xFFFFFF) as f32 / 16777216.0f32
}

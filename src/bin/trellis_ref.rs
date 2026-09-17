//! CPU reference for the trellis_encode kernel — mirrors the GLSL step-for-step so we can
//! byte-compare its packed output against the GPU's. If they match, the kernel is correct;
//! if not, this pinpoints whether the DP or the traceback/packing diverged.

const TILE: u32 = 32;
const RING: usize = (TILE * TILE) as usize; // 1024
const NSTATE: usize = 256;
const WIN: usize = 32;

// ── Exact mirrors of the GLSL helpers. ────────────────────────────────────────
fn mul1_decode(i: u32) -> f32 {
	// GLSL does this in 32-bit uint: (i * 0x83DCD12Du) & 0xFFFFu. Mirror the low-32 wrap.
	let hw = ((i as u64).wrapping_mul(0x83DCD12Du64) & 0xFFFF) as u32;
	let hw = hw + 0x6400u32;
	// unpackHalf2x16(hw).x — decode the low half of a packed half2.
	let bits = hw & 0xFFFFu32;
	half_to_f32(bits)
}

fn half_to_f32(h: u32) -> f32 {
	// IEEE 754 half → f32, matching GLSL unpackHalf2x16.
	let sign = ((h >> 11) & 0x1) as u32;
	let exp = (h >> 10) & 0x1F;
	let frac = h & 0x3FF;
	let f: f32;
	if exp == 0 {
		f = if frac == 0 {
			0.0
		} else {
			(frac as f32) * 2f32.powi(-14) // subnormal
		};
	} else if exp == 0x1F {
		f = if frac != 0 { f32::NAN } else { f32::INFINITY };
	} else {
		f = (frac as f32 + 1024.0) * 2f32.powi((exp as i32) - 18);
	}
	if sign == 1 { -f } else { f }
}

fn fp_sq_err(a: f32, b: f32) -> u32 {
	let d = (a - b).abs();
	// clamp(abs(d)*1024.0, 0.0, 65535.0) then uint()
	let m = ((d * 1024.0).clamp(0.0, 65535.0)) as u32;
	(m.wrapping_mul(m)) >> 4
}

// One tile: returns the packed words exactly as the kernel writes them to o.wdata[].
fn quantize_tile(data: &[f32], k: u32) -> Vec<u32> {
	let mask = if k > 0 { (1u32 << k) - 1 } else { 0 };
	let n_win_words = (WIN as u32 * k + 31) / 32; // ≤4 at K≤8

	// cost row: state 0 free, others pay 1.
	let mut cost = vec![1u32; NSTATE];
	cost[0] = 0;
	// bcost scratch + backpointer history for the current window (WIN rows × NSTATE).
	let mut bcost = vec![0u32; NSTATE];
	let mut bp: Vec<Vec<u8>> = vec![vec![0u8; NSTATE]; WIN];

	let mut win_words = vec![0u32; RING / WIN * 4]; // worst case words for the tile

	for w in 0..(RING / WIN) {
		// Forward pass over this window's 32 steps.
		for st in 0..WIN {
			let pos = w * WIN + st;
			let x = data[pos];
			// viterbi_step: each target t has two legal predecessors.
			for t in 0..NSTATE {
				let p0 = ((t as u32) << 1 | 0u32) & 0xFF;
				let p1 = ((t as u32) << 1 | 1u32) & 0xFF;
				let s0 = mul1_decode(p0 & mask);
				let s1 = mul1_decode(p1 & mask);
				let c0 = cost[p0 as usize].wrapping_add(fp_sq_err(x, s0));
				let c1 = cost[p1 as usize].wrapping_add(fp_sq_err(x, s1));
				if c1 < c0 {
					bcost[t] = c1;
					bp[st][t] = p1 as u8;
				} else {
					bcost[t] = c0;
					bp[st][t] = p0 as u8;
				}
			}
			std::mem::swap(&mut cost, &mut bcost);
		}

		// Traceback: find min-cost end state, walk back WIN steps.
		let mut min_c = cost[0];
		let mut min_idx = 0usize;
		for s in 1..NSTATE {
			if cost[s] < min_c { min_c = cost[s]; min_idx = s; }
		}
		let mut end_state = min_idx as u32;

		// symbol at step `step` = the state we were in at that step (before stepping back).
		let mut symbols = [0u32; WIN];
		for step in (0..WIN).rev() {
			symbols[step] = end_state & mask;
			end_state = bp[step][end_state as usize] as u32;
		}

		// Pack: symbol l's K bits → global bit position l*K .. l*K+K-1, within this window.
		for l in 0..WIN {
			let sym = symbols[l];
			for b in 0..k {
				if (sym >> b) & 1 != 0 {
					let bit_pos = l as u32 * k + b;
					let word_in_win = (bit_pos / 32) as usize;
					let bit_in_word = (bit_pos % 32) as usize;
					win_words[w * n_win_words as usize + word_in_win] |= 1u32 << bit_in_word;
				}
			}
		}
	}

	// The kernel writes total_packed_words = ceil(RING*K/32) words.
	let total = (RING as u32 * k + 31) / 32;
	win_words.truncate(total as usize);
	win_words
}

fn main() {
	let k: u32 = 4;
	// Tile 0's weights, exactly as make_ring_bytes produces them.
	let i = 0u32;
	let mut data = Vec::with_capacity(RING);
	for e in 0..RING as u32 {
		data.push(((e as f32 * 0.017 + (i as f32) * 0.001).sin() * 0.5));
	}
	let words = quantize_tile(&data, k);
	println!("CPU ref tile 0: {} packed words (K={})", words.len(), k);
	for (j, w) in words.iter().take(8).enumerate() {
		println!("  word[{j}] = 0x{w:08X}");
	}
	// Dump all as space-hex for easy diff against the GPU.
	let hex: Vec<String> = words.iter().map(|w| format!("{w:08X}")).collect();
	println!("FULL: {}", hex.join(" "));
}

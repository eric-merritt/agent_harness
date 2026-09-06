//! Sandbag read-path benchmarks.
//!
//! Not a decompression pipeline — there isn't one. A weight is reconstructed in
//! place from its own prefix, tail, sign bit and block scale, which is what the
//! GEMV kernel does inline. This measures that reconstruction, plus the index
//! parse and the offset derivation that locate a tensor without a stored table.

use agent_harness::models::format::*;
use agent_harness::models::quantize::*;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

const N: usize = 1 << 20;

fn synthetic_weights(n: usize) -> Vec<f32> {
	let mut s = 0x9E37_79B9_7F4A_7C15u64;
	let mut next = || {
		s ^= s << 13;
		s ^= s >> 7;
		s ^= s << 17;
		s
	};
	(0..n)
		.map(|_| {
			let u1 = (next() >> 11) as f32 / (1u64 << 53) as f32;
			let u2 = (next() >> 11) as f32 / (1u64 << 53) as f32;
			(-2.0 * (u1.max(1e-9)).ln()).sqrt() * (std::f32::consts::TAU * u2).cos() * 0.02
		})
		.collect()
}

/// Encoded payload plus the plane offsets a reader derives from elem_count.
struct Encoded {
	bytes: Vec<u8>,
	pairs_at: usize,
	signs_at: usize,
}

fn encoded(n: usize, digits: u32) -> Encoded {
	let vals = synthetic_weights(n);
	let threshold = calibrate(&vals, ScaleMethod::MaxAbs);
	Encoded {
		bytes: encode_sandbag(&vals, digits, threshold),
		pairs_at: sandbag_pairs_offset(n as u64) as usize,
		signs_at: sandbag_sign_offset(n as u64) as usize,
	}
}

/// Reconstruct every weight in a tensor.
fn decode_all(e: &Encoded, n: usize, digits: u32, out: &mut [f32]) {
	let block = SANDBAG_BLOCK_ELEMS as usize;
	for b in 0..n.div_ceil(block) {
		let scale = half::f16::from_le_bytes([e.bytes[b * 2], e.bytes[b * 2 + 1]]).to_f32();
		let lo = b * block;
		let hi = (lo + block).min(n);
		for i in lo..hi {
			let prefix = e.bytes[e.pairs_at + i * 2];
			let tail = e.bytes[e.pairs_at + i * 2 + 1];
			let word = u64::from_le_bytes(
				e.bytes[e.signs_at + (i / 64) * 8..e.signs_at + (i / 64) * 8 + 8]
					.try_into()
					.unwrap(),
			);
			let negative = (word >> (i % 64)) & 1 == 1;
			out[i] = decode_sandbag_weight(prefix, tail, negative, scale, digits);
		}
	}
}

fn bench_decode_weights(c: &mut Criterion) {
	let mut g = c.benchmark_group("sandbag/decode");
	g.throughput(Throughput::Elements(N as u64));
	let mut out = vec![0.0f32; N];
	for digits in [1u32, 2, 3] {
		let e = encoded(N, digits);
		g.bench_with_input(BenchmarkId::from_parameter(digits), &digits, |b, &d| {
			b.iter(|| decode_all(black_box(&e), N, d, &mut out));
		});
	}
	g.finish();
}

/// Just the sign plane — one bit per weight, read as u64 words.
fn bench_sign_plane(c: &mut Criterion) {
	let e = encoded(N, DEFAULT_TAIL_DIGITS);
	let mut g = c.benchmark_group("sandbag/sign_plane");
	g.throughput(Throughput::Elements(N as u64));
	g.bench_function("popcount", |b| {
		b.iter(|| {
			let mut negatives = 0u32;
			let words = N.div_ceil(64);
			for w in 0..words {
				let word = u64::from_le_bytes(
					e.bytes[e.signs_at + w * 8..e.signs_at + w * 8 + 8]
						.try_into()
						.unwrap(),
				);
				negatives += black_box(word).count_ones();
			}
			negatives
		});
	});
	g.finish();
}

/// Index parse and offset derivation — the work that replaces a stored offset
/// table. Cost scales with tensor count, not with model size.
fn bench_index_parse(c: &mut Criterion) {
	let mut g = c.benchmark_group("sandbag/index");
	for count in [64usize, 320, 1024] {
		let file = synthetic_sandbag_file(count);
		g.throughput(Throughput::Elements(count as u64));
		g.bench_with_input(BenchmarkId::from_parameter(count), &file, |b, f| {
			b.iter(|| SandbagReader::from_bytes(black_box(f)).unwrap());
		});
	}
	g.finish();
}

/// A header + index with `count` small tensors, and a payload sized to match.
fn synthetic_sandbag_file(count: usize) -> Vec<u8> {
	let elems: u64 = 256;
	let per = sandbag_tensor_bytes(elems);
	let header = SandbagHeader::new(count as u64, per * count as u64);

	let mut out = Vec::new();
	out.extend_from_slice(&header.magic.to_le_bytes());
	out.extend_from_slice(&header.version.to_le_bytes());
	out.extend_from_slice(&header.num_tensors.to_le_bytes());
	out.extend_from_slice(&header.total_data_bytes.to_le_bytes());

	for i in 0..count {
		let name = format!("blk.{i}.attn_q.weight");
		out.extend_from_slice(&(name.len() as u64).to_le_bytes());
		out.extend_from_slice(name.as_bytes());
		out.extend_from_slice(&2u64.to_le_bytes());
		out.extend_from_slice(&16u64.to_le_bytes());
		out.extend_from_slice(&16u64.to_le_bytes());
		out.push(GgmlType::Sandbag as u8);
	}
	out.resize(out.len() + (per * count as u64) as usize, 0);
	out
}

criterion_group!(
	benches,
	bench_decode_weights,
	bench_sign_plane,
	bench_index_parse
);
criterion_main!(benches);

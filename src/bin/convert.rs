// Standalone conversion tool: GGUF or Safetensors → CompressedTensor
//
// Usage:
//   cargo run --bin convert -- --gguf /path/to/model.gguf --out output/dir
//   cargo run --bin convert -- --safetensors-dir /path/to/model_dir/ --out output/dir
//   cargo run --bin convert -- --extract-meta /path/to/model.gguf --out output/dir

use std::fs;
use std::path::{Path, PathBuf};

fn main() {
	env_logger::Builder::from_env(
		env_logger::Env::default().default_filter_or("info,wgpu=warn,wgpu_core=warn,naga=warn"),
	)
	.init();

	let args: Vec<String> = std::env::args().collect();
	if args.len() < 2 {
		print_usage();
		std::process::exit(1);
	}

	let mut gguf_path: Option<String> = None;
	let mut safetensors_dir: Option<String> = None;
	let mut extract_meta: Option<String> = None;
	let mut out_dir: String = "output/converted".to_string();
	let mut prefix_digits: usize = 2;
	let mut truncate_rounds: usize = 2;
	let mut workers: usize = 24;

	let mut i = 1;
	while i < args.len() {
		match args[i].as_str() {
			"--gguf" => {
				gguf_path = Some(args[i + 1].clone());
				i += 2;
			}
			"--safetensors-dir" => {
				safetensors_dir = Some(args[i + 1].clone());
				i += 2;
			}
			"--extract-meta" => {
				extract_meta = Some(args[i + 1].clone());
				i += 2;
			}
			"--out" => {
				out_dir = args[i + 1].clone();
				i += 2;
			}
			"--prefix-digits" => {
				prefix_digits = args[i + 1].parse().unwrap_or(2);
				i += 2;
			}
			"--truncate-rounds" => {
				truncate_rounds = args[i + 1].parse().unwrap_or(2);
				i += 2;
			}
			"--workers" => {
				workers = args[i + 1].parse().unwrap_or(24);
				i += 2;
			}
			"--help" | "-h" => {
				print_usage();
				return;
			}
			_ => {
				eprintln!("Unknown arg: {}", args[i]);
				print_usage();
				std::process::exit(1);
			}
		}
	}

	let out_path = Path::new(&out_dir);

	if let Some(gguf) = extract_meta {
		// The GGUF key-value reader lives behind `models::convert::parse_gguf_header`,
		// which is still a stub (it returns an empty metadata map). Emitting a
		// config.json from it would silently write an all-zero model config, so
		// refuse rather than produce a plausible-looking wrong file.
		eprintln!(
			"--extract-meta is unavailable: the GGUF metadata parser is not implemented \
			 (models::convert::parse_gguf_header returns no key-value pairs)."
		);
		eprintln!("  requested: {}", gguf);
		std::process::exit(1);
	}

	if let Some(gguf) = gguf_path {
		eprintln!("Converting GGUF: {}", gguf);
		eprintln!(
			"  prefix_digits={}, truncate_rounds={}, workers={}",
			prefix_digits, truncate_rounds, workers
		);
		let dst = out_path.join("model.sandbag");
		agent_harness::models::convert::convert_gguf_to_sandbag(Path::new(&gguf), &dst)
			.expect("GGUF conversion failed");
		print_summary(&dst);
	} else if let Some(st_dir) = safetensors_dir {
		let dir = Path::new(&st_dir);
		let mut shards: Vec<PathBuf> = fs::read_dir(dir)
			.expect("can't read model dir")
			.filter_map(|e| e.ok())
			.map(|e| e.path())
			.filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
			.collect();
		if shards.is_empty() {
			eprintln!("No .safetensors files found in {}", st_dir);
			std::process::exit(1);
		}
		shards.sort();
		println!("Found {} shards", shards.len());
		// One sandbag file per shard, named after the shard. Shard order is
		// preserved so tensors keep their original positions.
		for shard in &shards {
			let stem = shard.file_stem().unwrap_or_default().to_string_lossy();
			let dst = out_path.join(format!("{}.sandbag", stem));
			eprintln!("Converting {} → {}", shard.display(), dst.display());
			agent_harness::models::convert::convert_safetensors_to_sandbag(shard, &dst)
				.expect("safetensors conversion failed");
			print_summary(&dst);
		}
	} else {
		eprintln!("Must specify --gguf or --safetensors-dir");
		print_usage();
		std::process::exit(1);
	}
}

fn print_usage() {
	eprintln!("Usage: convert [OPTIONS]");
	eprintln!();
	eprintln!("Options:");
	eprintln!("  --gguf <path>              Path to a .gguf file");
	eprintln!("  --safetensors-dir <dir>    Directory containing .safetensors files");
	eprintln!(
		"  --extract-meta <path>      Extract config.json + tokenizer.json from GGUF to --out dir"
	);
	eprintln!("  --out <dir>                Output directory (default: output/converted)");
	eprintln!("  --prefix-digits <n>        Prefix digits for compression (default: 2)");
	eprintln!("  --truncate-rounds <n>      Tail truncation rounds (default: 2)");
	eprintln!("  --workers <n>              Number of worker threads (default: 24)");
	eprintln!("  --help, -h                 Show this help");
}

/// Report what actually landed on disk by reading the sandbag file back.
fn print_summary(dst: &Path) {
	use agent_harness::models::format::SandbagReader;

	let reader = match SandbagReader::from_path(dst) {
		Ok(r) => r,
		Err(e) => {
			eprintln!("Wrote {} but could not read it back: {}", dst.display(), e);
			return;
		}
	};
	let sand_mb = reader.header().total_data_bytes as f64 / 1_048_576.0;
	println!();
	println!("Output:   {}", dst.display());
	println!("Tensors:  {}", reader.header().num_tensors);
	println!("Sandbag:  {:.1} MB", sand_mb);
}

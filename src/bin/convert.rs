// Standalone conversion tool: GGUF or Safetensors → DedupCountTensor
//
// Usage:
//   cargo run --bin convert -- --gguf /path/to/model.gguf --out output/dir
//   cargo run --bin convert -- --safetensors-dir /path/to/model_dir/ --out output/dir
//   cargo run --bin convert -- --extract-meta /path/to/model.gguf --out output/dir

use std::path::{Path, PathBuf};
use std::fs;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,wgpu=warn,wgpu_core=warn,naga=warn")).init();

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
            "--gguf" => { gguf_path = Some(args[i+1].clone()); i += 2; }
            "--safetensors-dir" => { safetensors_dir = Some(args[i+1].clone()); i += 2; }
            "--extract-meta" => { extract_meta = Some(args[i+1].clone()); i += 2; }
            "--out" => { out_dir = args[i+1].clone(); i += 2; }
            "--prefix-digits" => { prefix_digits = args[i+1].parse().unwrap_or(2); i += 2; }
            "--truncate-rounds" => { truncate_rounds = args[i+1].parse().unwrap_or(2); i += 2; }
            "--workers" => { workers = args[i+1].parse().unwrap_or(24); i += 2; }
            "--help" | "-h" => { print_usage(); return; }
            _ => { eprintln!("Unknown arg: {}", args[i]); print_usage(); std::process::exit(1); }
        }
    }

    let out_path = Path::new(&out_dir);

    if let Some(gguf) = extract_meta {
        println!("Extracting metadata from GGUF: {}", gguf);
        let gguf_file = agent_harness::models::formats::gguf::GGUFFile::from_file(Path::new(&gguf))
            .expect("Failed to parse GGUF");
        let config = agent_harness::inference::config::ModelConfig::from_gguf(&gguf_file);
        config.to_file(out_path).expect("Failed to write config.json");
        agent_harness::inference::tokenizer::Tokenizer::extract_to_file(&gguf_file, out_path)
            .expect("Failed to write tokenizer.json");
        println!("Written config.json and tokenizer.json to {}", out_path.display());
        return;
    }

    if let Some(gguf) = gguf_path {
        eprintln!("Converting GGUF: {}", gguf);
        eprintln!("  prefix_digits={}, truncate_rounds={}, workers={}", prefix_digits, truncate_rounds, workers);
        let stats = agent_harness::models::convert::gguf::convert_gguf(
            Path::new(&gguf), out_path, prefix_digits, truncate_rounds, workers,
        ).expect("GGUF conversion failed");
        print_summary(&stats);
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
        println!("Found {} shards, {} workers", shards.len(), workers);
        let stats = agent_harness::models::convert::safetensors::convert_safetensors_parallel(
            &shards, out_path, prefix_digits, truncate_rounds, workers,
        ).expect("safetensors conversion failed");
        print_summary(&stats);
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
    eprintln!("  --extract-meta <path>      Extract config.json + tokenizer.json from GGUF to --out dir");
    eprintln!("  --out <dir>                Output directory (default: output/converted)");
    eprintln!("  --prefix-digits <n>        Prefix digits for compression (default: 2)");
    eprintln!("  --truncate-rounds <n>      Tail truncation rounds (default: 2)");
    eprintln!("  --workers <n>              Number of worker threads (default: 24)");
    eprintln!("  --help, -h                 Show this help");
}

fn print_summary(stats: &agent_harness::models::convert::common::ConversionStats) {
    let orig_mb = stats.total_original_bytes as f64 / 1_048_576.0;
    let core_mb = stats.total_core_bytes as f64 / 1_048_576.0;
    let sand_mb = stats.total_sandbag_bytes as f64 / 1_048_576.0;
    let ratio = if stats.total_core_bytes > 0 {
        stats.total_original_bytes as f64 / stats.total_core_bytes as f64
    } else { 1.0 };
    println!();
    println!("Model: {}", stats.model_name);
    println!("Tensors: {}", stats.tensor_count);
    println!("Original: {:.1} MB", orig_mb);
    println!("Core:     {:.1} MB", core_mb);
    println!("Sandbag:  {:.1} MB", sand_mb);
    println!("Ratio:    {:.2}x", ratio);
    for t in &stats.tensors {
        if t.full_precision {
            println!("  [FP]  {}", t.name);
        }
    }
}

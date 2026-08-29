#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use crate::inference::KvCache;
use agent_harness::*;
use std::env;
use std::io::Write;
use std::path::Path;
use std::process::Command;

/// File wrapper that auto-flushes after every write — env_logger's Target::Pipe
/// uses buffered File by default, which delays log visibility until the buffer fills.
struct AutoFlushFile(std::fs::File);
impl Write for AutoFlushFile {
	fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
		let n = self.0.write(buf)?;
		let _ = self.0.flush();
		Ok(n)
	}
	fn flush(&mut self) -> std::io::Result<()> {
		self.0.flush()
	}
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {

	    // Tell Cargo to rerun this script if the shader changes
    println!("cargo:rerun-if-changed=src/models/dedupe/quantize_gemv.comp");

    let status = Command::new("glslangValidator")
        .args(&[
            "-V", 
            "-S", "comp", 
            "src/models/dedupe/quantize_gemv.comp", 
            "-o", "src/models/dedupe/quantize_gemv.spv"
        ])
        .status()
        .expect("Failed to execute glslangValidator");

    if !status.success() {
        panic!("Shader compilation failed!");
    }

	// ── Logging: always on at INFO, DEBUG with --verbose ──
	// Write to a log file so logs don't corrupt the TUI alt screen.
	let verbose = std::env::args().any(|a| a == "--verbose" || a == "-v");
	let level = if verbose { "debug" } else { "info" };
	let log_dir = std::env::var("LOG_DIR").unwrap_or_else(|_| "logs".to_string());
	std::fs::create_dir_all(&log_dir).ok();
	let log_file = std::fs::OpenOptions::new()
		.create(true)
		.append(true)
		.open(format!("{}/agent_harness.log", log_dir))
		.unwrap_or_else(|e| {
			// Fallback to stderr if file can't be opened
			eprintln!("Failed to open log file, falling back to stderr: {}", e);
			std::fs::OpenOptions::new()
				.create(true)
				.append(true)
				.open("/dev/stderr")
				.unwrap()
		});

	unsafe {
		env::set_var(
			"DATABASE_URL",
			"postgres://ermer:ermer@localhost/agent_harness",
		)
	};
	env_logger::Builder::from_env(
		env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"),
	)
	.format_timestamp_millis()
	.target(env_logger::Target::Pipe(Box::new(AutoFlushFile(log_file))))
	.init();

	// Redirect panic messages to the log file too — otherwise they go to
	// stderr and corrupt the TUI alt screen.
	std::panic::set_hook(Box::new(|info| {
		log::error!("PANIC: {}", info);
	}));

	log::info!("Agent Harness starting (log level: {})", level);

	// ── Optional DB connection ──
	let db = if let Ok(url) = std::env::var("DATABASE_URL") {
		match sqlx::PgPool::connect(&url).await {
			Ok(pool) => {
				let db = database::postgres::Database::new(pool).await;
				if let Err(e) = db.init().await {
					log::warn!("DB init: {}", e);
				}
				log::info!("Database connected");
				Some(db)
			}
			Err(e) => {
				log::warn!("DB connection failed (running in-memory): {}", e);
				None
			}
		}
	} else {
		log::info!("DATABASE_URL not set — running in-memory only");
		None
	};

	// ── Auto-detect compressed model ──
	// Pass the path to App — the model loads in a background thread with a loading modal.
	let model_path = find_compressed_model_path();

	log::info!("Starting UI");
	ui_ux::app::App::new(db, model_path).run();

	log::info!("Agent Harness shutting down");
	Ok(())
}

/// Scan for a compressed model directory.
/// Checks MODEL_DIR env var first, then scans output/ for manifest.json.
/// Returns the path only — the model is loaded in a background thread by App.
fn find_compressed_model_path() -> Option<std::path::PathBuf> {
	// 1. MODEL_DIR env var
	if let Ok(dir) = std::env::var("MODEL_DIR") {
		let path = Path::new(&dir);
		if path.join("manifest.json").exists() {
			log::info!("Found model from MODEL_DIR: {}", dir);
			return Some(path.to_path_buf());
		}
	}

	// 2. Scan output/ subdirectories for the most recent manifest.json
	let output_dir = Path::new("output");
	if output_dir.exists() {
		let mut candidates: Vec<(std::time::SystemTime, std::path::PathBuf)> = Vec::new();
		if let Ok(entries) = std::fs::read_dir(output_dir) {
			for entry in entries.flatten() {
				let path = entry.path();
				if path.join("manifest.json").exists() {
					if let Ok(meta) = entry.metadata() {
						if let Ok(modified) = meta.modified() {
							candidates.push((modified, path));
						}
					}
				}
			}
		}
		if !candidates.is_empty() {
			candidates.sort_by(|a, b| b.0.cmp(&a.0));
			let newest = &candidates[0].1;
			log::info!("Auto-detected model: {}", newest.display());
			return Some(newest.clone());
		}
	}

	log::warn!("No compressed model found — set MODEL_DIR or place one in output/");
	None
}

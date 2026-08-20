fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args: Vec<String> = std::env::args().collect();
    let default_dir = "output/Qwen3.5-9B-fixed".to_string();
    let dir = std::path::Path::new(args.get(1).unwrap_or(&default_dir));
    let mut loader = agent_harness::models::convert::ModelLoader::open(dir)
        .expect("Failed to open model");

    let tensors = vec![
        "token_embd.weight",
        "output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ssm_a",
    ];
    for name in &tensors {
        let start = std::time::Instant::now();
        match loader.decompress_tensor(name) {
            Ok(w) => {
                let non_zero = w.iter().filter(|&&v| v != 0.0).count();
                let min = w.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean = w.iter().sum::<f32>() / w.len() as f32;
                log::info!("  {} → {} elements in {:.3}s | min={:.6} max={:.6} mean={:.6} nonzero={}/{}",
                    name, w.len(), start.elapsed().as_secs_f64(),
                    min, max, mean, non_zero, w.len());
                // Print first 5 values
                let preview: Vec<String> = w.iter().take(5).map(|v| format!("{:.6}", v)).collect();
                log::info!("    first 5: [{}]", preview.join(", "));
            }
            Err(e) => log::error!("  {} → FAILED: {}", name, e),
        }
    }

    // Test tokenizer
    let tok_dir = std::path::Path::new(args.get(1).unwrap_or(&default_dir));
    match agent_harness::inference::tokenizer::Tokenizer::from_dir(tok_dir) {
        Ok(tok) => {
            log::info!("Tokenizer: {} tokens", tok.vocab.len());
            let tokens = tok.encode("Hi there.");
            log::info!("  'Hi there.' → {} tokens: {:?}", tokens.len(), tokens);
            let decoded = tok.decode(&tokens);
            log::info!("  decoded back: '{}'", decoded);
            // Check first few vocab entries
            for i in 0..5 {
                log::info!("  vocab[{}] = {:?}", i, tok.vocab.get(i));
            }
        }
        Err(e) => log::error!("Tokenizer failed: {}", e),
    }
}

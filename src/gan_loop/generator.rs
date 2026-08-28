// Generator agent — produces output for each pending/rejected task.
// Calls the LLM with the task description, goal context, and any evaluator feedback.

use super::state::{GANLoopState, TaskStatus};

const GENERATOR_SYSTEM: &str = "You are a generating agent. Your job is to produce a concrete, complete \
     result for the given task. The overall goal is provided for context. \
     Output your result as plain text. Be specific and actionable.";

/// Run one generation pass: attempt every task that needs generation.
pub async fn generate(state: &mut GANLoopState) {
	let endpoint =
		std::env::var("LLM_ENDPOINT").unwrap_or_else(|_| "http://localhost:8842".to_string());
	let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "default".to_string());
	let url = format!("{}/v1/chat/completions", endpoint);

	let client = match reqwest::Client::builder()
		.timeout(std::time::Duration::from_secs(60))
		.build()
	{
		Ok(c) => c,
		Err(_) => return,
	};

	for task in &mut state.tasks {
		if !task.needs_generation() {
			continue;
		}

		// Build the user prompt: goal + task + any evaluator feedback
		let mut user_prompt = format!("GOAL: {}\n\nTASK: {}", state.goal, task.description);
		if let Some(ref notes) = task.evaluator_notes {
			user_prompt.push_str(&format!("\n\nPREVIOUS FEEDBACK: {}", notes));
		}

		let body = serde_json::json!({
			"model": model,
			"messages": [
				{ "role": "system", "content": GENERATOR_SYSTEM },
				{ "role": "user", "content": &user_prompt },
			],
		});

		let generation_text: Option<String> = async {
			let resp = client
				.post(&url)
				.header("Content-Type", "application/json")
				.json(&body)
				.send()
				.await
				.ok()?;
			if !resp.status().is_success() {
				return None;
			}
			let text = resp.text().await.ok()?;
			let val: serde_json::Value = serde_json::from_str(&text).ok()?;
			let content = val["choices"][0]["message"]["content"].as_str()?;
			Some(content.to_string())
		}
		.await;

		match generation_text {
			Some(content) => {
				task.generation = Some(content);
				task.status = TaskStatus::AwaitingReview;
				task.evaluator_notes = None;
				task.iteration_count += 1;
				log::info!(
					"[Generator] Task '{}' — iteration {} complete",
					task.description,
					task.iteration_count
				);
			}
			None => {
				// LLM unavailable — store a placeholder so the loop doesn't stall
				task.generation = Some("[LLM unavailable — generation skipped]".to_string());
				task.status = TaskStatus::AwaitingReview;
				task.evaluator_notes = None;
				task.iteration_count += 1;
				log::warn!(
					"[Generator] Task '{}' — LLM call failed, using placeholder",
					task.description
				);
			}
		}
	}
}

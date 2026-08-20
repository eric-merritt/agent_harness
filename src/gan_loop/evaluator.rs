// Evaluator agent — reviews each generated task and marks Complete or Rejected.
// Calls the LLM with the task + generation and asks for a binary verdict + notes.

use super::state::{GANLoopState, TaskStatus};

const EVALUATOR_SYSTEM: &str =
    "You are an evaluating agent. Review the generation for the given task. \
     Output ONLY a JSON object with two fields: \
     {\"complete\": true|false, \"notes\": \"...\"}. \
     If complete is true, the task is done. If false, notes must explain what's missing.";

/// Run one evaluation pass: review every task awaiting review.
pub async fn evaluate(state: &mut GANLoopState) {
    let endpoint = std::env::var("LLM_ENDPOINT").unwrap_or_else(|_| "http://localhost:8842".to_string());
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
        if !task.needs_evaluation() {
            continue;
        }

        let generation = task.generation.as_deref().unwrap_or("No output produced.");
        let user_prompt = format!(
            "TASK: {}\n\nGENERATION:\n{}",
            task.description, generation
        );

        let body = serde_json::json!({
            "model": model,
            "messages": [
                { "role": "system", "content": EVALUATOR_SYSTEM },
                { "role": "user", "content": &user_prompt },
            ],
        });

        let verdict = async {
            let resp = client.post(&url)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .ok()?;
            if !resp.status().is_success() { return None; }
            let text = resp.text().await.ok()?;
            let val: serde_json::Value = serde_json::from_str(&text).ok()?;
            let content = val["choices"][0]["message"]["content"].as_str()?;

            // Parse JSON verdict
            let verdict: serde_json::Value = serde_json::from_str(content).ok()?;
            let complete = verdict["complete"].as_bool()?;
            let notes = verdict["notes"].as_str().map(|s| s.to_string());
            Some((complete, notes))
        }.await;

        match verdict {
            Some((true, _)) => {
                task.status = TaskStatus::Complete;
                task.evaluator_notes = None;
                log::info!("[Evaluator] Task '{}' — COMPLETE", task.description);
            }
            Some((false, notes)) => {
                task.status = TaskStatus::Rejected { notes: notes.clone().unwrap_or_default() };
                task.evaluator_notes = notes;
                log::warn!("[Evaluator] Task '{}' — REJECTED", task.description);
            }
            None => {
                // LLM unavailable — optimistic default to keep the loop moving
                task.status = TaskStatus::Complete;
                task.evaluator_notes = Some("[Evaluator LLM unavailable — marked complete by default]".to_string());
                log::warn!("[Evaluator] Task '{}' — LLM unavailable, defaulting to complete", task.description);
            }
        }
    }
}

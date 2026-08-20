// Planner agent — decomposes a high-level goal into a list of concrete tasks.
// Calls the LLM with a planning system prompt to get structured task output.

use super::state::{TaskItem, GANLoopState};

const PLANNER_SYSTEM: &str =
    "You are a planning agent. Break the user's goal into a minimal set of \
     independent, actionable tasks. Output ONLY a JSON array of task description \
     strings, one per task. Example: [\"Set up database\", \"Write API layer\", \"Add tests\"]. \
     Keep tasks specific and achievable.";

/// Decompose the goal into tasks and store them in the loop state.
/// Uses the LLM if available; falls back to a single-task passthrough.
pub async fn plan(state: &mut GANLoopState) {
    if !state.tasks.is_empty() {
        return; // already planned
    }

    // Try LLM-based decomposition
    let llm_tasks = async {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .ok()?;

        let endpoint = std::env::var("LLM_ENDPOINT").unwrap_or_else(|_| "http://localhost:8842".to_string());
        let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "default".to_string());
        let url = format!("{}/v1/chat/completions", endpoint);

        let body = serde_json::json!({
            "model": model,
            "messages": [
                { "role": "system", "content": PLANNER_SYSTEM },
                { "role": "user", "content": &state.goal },
            ],
        });

        let resp = client.post(&url)
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

        // Extract assistant message content
        let content = val["choices"][0]["message"]["content"].as_str()?;

        // Try to parse as JSON array of strings
        let tasks: Vec<String> = serde_json::from_str(content).ok()?;
        Some(tasks)
    }.await;

    match llm_tasks {
        Some(tasks) if !tasks.is_empty() => {
            for desc in tasks {
                state.tasks.push(TaskItem::new(desc));
            }
            log::info!("[Planner] Decomposed goal into {} tasks", state.tasks.len());
        }
        _ => {
            // Fallback: single task = the goal itself
            state.tasks.push(TaskItem::new(state.goal.clone()));
            log::warn!("[Planner] LLM unavailable — using single-task fallback");
        }
    }
}

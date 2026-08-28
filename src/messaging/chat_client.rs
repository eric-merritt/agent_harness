// OpenAI-compatible chat completions client.
//
// Sends a message list to the /v1/chat/completions endpoint
// and returns the assistant's response text.
// Supports both non-streaming and streaming (SSE) responses.

use anyhow::{Context, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

/// A single message in the conversation history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessageInput {
	pub role: String,
	pub content: String,
}

/// Minimal response shape — only the fields we need.
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
	choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
	message: ChatMessageInput,
}

pub struct ChatApiClient {
	endpoint: String,
	model: String,
	client: reqwest::Client,
}

impl ChatApiClient {
	pub fn new(endpoint: &str, model: &str) -> Self {
		let client = reqwest::Client::builder()
			.timeout(std::time::Duration::from_secs(30))
			.build()
			.unwrap_or_else(|e| {
				log::error!(
					"failed to build reqwest client with custom timeout, using default: {}",
					e
				);
				reqwest::Client::new()
			});
		log::debug!(
			"chat client created: endpoint={}, model={}",
			endpoint,
			model
		);
		Self {
			endpoint: endpoint.to_string(),
			model: model.to_string(),
			client,
		}
	}

	/// Send the full conversation and return the assistant reply as plain text.
	pub async fn chat(&self, messages: Vec<ChatMessageInput>) -> Result<String> {
		let url = format!("{}/v1/chat/completions", self.endpoint);

		let body = serde_json::json!({
			"model": self.model,
			"messages": messages,
		});

		log::debug!("sending chat request to {} (model={})", url, self.model);

		let resp = self
			.client
			.post(&url)
			.header("Content-Type", "application/json")
			.json(&body)
			.send()
			.await
			.context("POST to chat completions failed")?;

		let status = resp.status();
		let body_text = resp.text().await.context("read response body")?;

		if !status.is_success() {
			log::warn!("chat completions returned HTTP {} from {}", status, url);
			anyhow::bail!(
				"HTTP {} from {}: {}",
				status,
				url,
				body_text.chars().take(300).collect::<String>()
			);
		}

		let response: ChatCompletionResponse =
			serde_json::from_str(&body_text).context("parse chat completion response")?;

		response
			.choices
			.into_iter()
			.next()
			.and_then(|c| Some(c.message.content))
			.ok_or_else(|| {
				log::error!("no choices in chat completion response");
				anyhow::anyhow!("no choices in response")
			})
	}

	/// Send the full conversation with `stream: true` and yield delta text chunks
	/// via an SSE (server-sent events) response.
	///
	/// `on_chunk` is called for every non-empty `content` delta.
	/// `on_done` is called when the stream finishes (or on error).
	/// Returns the full concatenated response on success.
	pub async fn chat_stream<F, G>(
		&self,
		messages: Vec<ChatMessageInput>,
		on_chunk: F,
		on_done: G,
	) -> Result<String>
	where
		F: Fn(&str) + Send + Sync,
		G: Fn(Option<anyhow::Error>) + Send + Sync,
	{
		let url = format!("{}/v1/chat/completions", self.endpoint);

		let body = serde_json::json!({
			"model": self.model,
			"messages": messages,
			"stream": true,
		});

		log::debug!("starting stream to {} (model={})", url, self.model);

		let resp = self
			.client
			.post(&url)
			.header("Content-Type", "application/json")
			.json(&body)
			.send()
			.await
			.context("POST to chat completions (stream) failed")?;

		let status = resp.status();
		if !status.is_success() {
			log::warn!(
				"stream chat completions returned HTTP {} from {}",
				status,
				url
			);
			let body_text = resp.text().await.unwrap_or_default();
			on_done(Some(anyhow::anyhow!(
				"HTTP {} from {}: {}",
				status,
				url,
				body_text.chars().take(300).collect::<String>()
			)));
			anyhow::bail!(
				"HTTP {} from {}: {}",
				status,
				url,
				body_text.chars().take(300).collect::<String>()
			);
		}

		let mut full_response = String::new();
		let mut stream = resp.bytes_stream();

		// Accumulate data across chunks (a single SSE event may be split)
		let mut buffer = String::new();

		while let Some(result) = stream.next().await {
			match result {
				Ok(chunk_bytes) => {
					let text = String::from_utf8_lossy(&chunk_bytes);
					buffer.push_str(&text);

					// Process complete lines
					loop {
						match buffer.find('\n') {
							Some(pos) => {
								let line: String = buffer.drain(..pos).collect();
								let line = line.trim();

								// SSE: skip comments
								if line.starts_with(':') {
									continue;
								}

								// End of event
								if line == "data: [DONE]" || line.is_empty() {
									continue;
								}

								if let Some(json) = line.strip_prefix("data: ") {
									if let Ok(parsed) =
										serde_json::from_str::<serde_json::Value>(json)
									{
										// Extract delta.content from choices[0].delta.content
										if let Some(content) = parsed
											.get("choices")
											.and_then(|c| c.as_array())
											.and_then(|arr| arr.first())
											.and_then(|c| c.get("delta"))
											.and_then(|d| d.get("content"))
											.and_then(|c| c.as_str())
										{
											if !content.is_empty() {
												full_response.push_str(content);
												on_chunk(content);
											}
										}
									}
								}
							}
							None => break,
						}
					}
				}
				Err(e) => {
					log::error!("stream error: {}", e);
					on_done(Some(anyhow::anyhow!("Stream error: {}", e)));
					return Err(e.into());
				}
			}
		}

		log::debug!(
			"stream completed ({} chars total)",
			full_response.chars().count()
		);
		on_done(None);
		Ok(full_response)
	}
}

impl Clone for ChatApiClient {
	fn clone(&self) -> Self {
		Self {
			endpoint: self.endpoint.clone(),
			model: self.model.clone(),
			client: self.client.clone(),
		}
	}
}

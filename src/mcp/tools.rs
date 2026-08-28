// Register the two MCP tools as Rust Tool trait implementations.

use async_trait::async_trait;
use serde_json::Value;

use super::client::McpClient;
use crate::tools::tool::{Tool, ToolContext, ToolResult};

/// Tool: "mcp_initialize" — connect to an MCP server and list its tools.
pub struct McpInitializeTool {
	client: McpClient,
}

impl McpInitializeTool {
	pub fn new() -> Self {
		Self {
			client: McpClient::default(),
		}
	}

	pub fn with_default_url(client: McpClient) -> Self {
		Self { client }
	}
}

#[async_trait]
impl Tool for McpInitializeTool {
	fn name(&self) -> &str {
		"mcp_initialize"
	}

	fn description(&self) -> &str {
		"Connect to an MCP server and list its available tools with their schemas."
	}

	async fn call(&self, _ctx: &ToolContext, params: &Value) -> ToolResult {
		let url = params.get("url").and_then(|v| v.as_str());
		log::debug!("mcp_initialize — url: {:?}", url);

		match self.client.connect(url).await {
			Ok(tools) => {
				log::info!(
					"MCP initialize succeeded — {} tool(s) discovered",
					tools.len()
				);
				let serialized: Vec<Value> = tools
					.iter()
					.map(|t| {
						serde_json::json!({
							"name": t.name,
							"description": t.description,
							"inputSchema": t.input_schema,
						})
					})
					.collect();
				ToolResult::ok_with_data(
					format!("Found {} tool(s) on MCP server", tools.len()),
					serde_json::json!({ "tools": serialized }),
				)
			}
			Err(e) => {
				log::warn!("MCP initialize failed: {}", e);
				ToolResult::err(format!("Failed to connect to MCP server: {}", e))
			}
		}
	}
}

/// Tool: "mcp_call" — call a specific tool on an MCP server.
pub struct McpCallTool {
	client: McpClient,
}

impl McpCallTool {
	pub fn new() -> Self {
		Self {
			client: McpClient::default(),
		}
	}
}

#[async_trait]
impl Tool for McpCallTool {
	fn name(&self) -> &str {
		"mcp_call"
	}

	fn description(&self) -> &str {
		"Call a specific tool on an MCP server with parameters. If url is omitted, defaults to the local tool server."
	}

	async fn call(&self, _ctx: &ToolContext, params: &Value) -> ToolResult {
		let url = params.get("url").and_then(|v| v.as_str());
		let tool_name = match params.get("tool_name").and_then(|v| v.as_str()) {
			Some(n) => n,
			None => return ToolResult::err("Missing required parameter: tool_name"),
		};
		log::debug!("mcp_call — tool: '{}' url: {:?}", tool_name, url);

		// Accept parameters as either a JSON string or a literal object.
		let tool_params: Value = match params.get("parameters") {
			Some(p) if p.is_string() => match serde_json::from_str(p.as_str().unwrap()) {
				Ok(v) => v,
				Err(e) => {
					log::warn!("Failed to parse MCP tool parameters: {}", e);
					return ToolResult::err(format!("Failed to parse parameters: {}", e));
				}
			},
			Some(p) => p.clone(),
			None => Value::Object(Default::default()),
		};

		log::info!("Calling MCP tool '{}'", tool_name);
		match self.client.call_tool(url, tool_name, &tool_params).await {
			Ok(result) => {
				log::info!("MCP tool '{}' call succeeded", tool_name);
				ToolResult::ok_with_data("Tool call succeeded", result)
			}
			Err(e) => {
				log::warn!("MCP tool '{}' call failed: {}", tool_name, e);
				ToolResult::err(format!("MCP tool call failed: {}", e))
			}
		}
	}
}

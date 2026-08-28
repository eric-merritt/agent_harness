// Execution sandbox — enforces timeout and resource limits on tool calls.
//
// This is a minimal wrapper.  In a full implementation this would use
// seccomp/cgroups/namespace isolation; for now it provides async timeout.

use serde_json::Value;
use std::time::Duration;

use super::tool::{Tool, ToolContext, ToolResult};

/// Default per-tool execution timeout.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Call a tool inside a timeout guard.
pub async fn run_with_timeout(
	tool: &dyn Tool,
	ctx: &ToolContext,
	params: &Value,
	timeout_secs: Option<u64>,
) -> ToolResult {
	let timeout = Duration::from_secs(timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS));
	log::debug!(
		"Running tool '{}' with {}s timeout",
		tool.name(),
		timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS)
	);
	match tokio::time::timeout(timeout, tool.call(ctx, params)).await {
		Ok(result) => {
			log::debug!("Tool '{}' completed within timeout", tool.name());
			result
		}
		Err(_) => {
			log::warn!(
				"Tool '{}' exceeded {}s timeout",
				tool.name(),
				timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS)
			);
			ToolResult::err(format!(
				"Tool '{}' exceeded {}s timeout",
				tool.name(),
				timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS)
			))
		}
	}
}

/// Convenience: call with default timeout.
pub async fn run(tool: &dyn Tool, ctx: &ToolContext, params: &Value) -> ToolResult {
	run_with_timeout(tool, ctx, params, None).await
}

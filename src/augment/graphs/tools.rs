use serde::{Deserialize, Serialize};

/// Relationship type between tool nodes in the graph.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolRelationship {
	Uses,
	Provides,
	Requires,
	BelongsTo,
	WhenToUse,
}

/// Result returned by a tool invocation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolResult {
	/// Whether the tool completed successfully.
	pub success: bool,
	/// Human-readable output or error message.
	pub message: String,
	/// Optional structured output payload.
	pub output: Option<Vec<u8>>,
}

impl ToolResult {
	/// Create a successful result with an optional payload.
	pub fn ok(message: impl Into<String>) -> Self {
		Self {
			success: true,
			message: message.into(),
			output: None,
		}
	}

	/// Create a successful result with a binary payload.
	pub fn with_output(message: impl Into<String>, output: Vec<u8>) -> Self {
		Self {
			success: true,
			message: message.into(),
			output: Some(output),
		}
	}

	/// Create a failure result.
	pub fn err(message: impl Into<String>) -> Self {
		Self {
			success: false,
			message: message.into(),
			output: None,
		}
	}
}

pub trait Tool {
	fn call(&self) -> ToolResult;
}

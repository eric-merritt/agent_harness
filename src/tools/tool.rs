// Tool trait and result types.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Errors that can occur during tool execution.
#[derive(Debug, Serialize, Deserialize)]
pub enum ToolError {
    /// The tool could not find the target resource.
    NotFound(String),
    /// The caller lacks permission.
    PermissionDenied(String),
    /// A network or I/O error.
    Io(String),
    /// The input parameters are malformed.
    InvalidParams(String),
    /// The tool timed out.
    Timeout(String),
    /// Some other error.
    Other(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::NotFound(msg) => write!(f, "Not found: {}", msg),
            ToolError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            ToolError::Io(msg) => write!(f, "I/O error: {}", msg),
            ToolError::InvalidParams(msg) => write!(f, "Invalid params: {}", msg),
            ToolError::Timeout(msg) => write!(f, "Timeout: {}", msg),
            ToolError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for ToolError {}

impl From<std::io::Error> for ToolError {
    fn from(e: std::io::Error) -> Self {
        ToolError::Io(e.to_string())
    }
}

/// The result of executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    /// Human-readable output on success.
    pub output: String,
    /// Structured payload (optional).
    pub data: Option<Value>,
    /// Error message on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            data: None,
            error: None,
        }
    }

    pub fn ok_with_data(output: impl Into<String>, data: Value) -> Self {
        Self {
            success: true,
            output: output.into(),
            data: Some(data),
            error: None,
        }
    }

    pub fn err(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            output: String::new(),
            data: None,
            error: Some(msg.into()),
        }
    }
}

/// Context passed into a tool call — environment, workspace, etc.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ToolContext {
    /// Working directory for relative paths.
    pub workspace: String,
    /// Arbitrary key-value environment the caller wants to expose.
    pub env: std::collections::HashMap<String, String>,
}

impl ToolContext {
    pub fn new(workspace: impl Into<String>) -> Self {
        Self {
            workspace: workspace.into(),
            env: std::collections::HashMap::new(),
        }
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }
}

/// Any tool that can be called by the agent.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn call(&self, ctx: &ToolContext, params: &Value) -> ToolResult;
}



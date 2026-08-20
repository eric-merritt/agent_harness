// MCP client — initializes a connection, lists tools, calls tools.
//
// Transport: Streamable HTTP (POST with SSE responses) per MCP spec.

use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use anyhow::{Context, Result};

/// A tool advertised by an MCP server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Value>,
}

/// Per-server session state.
#[derive(Debug)]
pub struct ServerSession {
    /// Normalised URL (trailing slash ensured).
    pub url: String,
    /// Whether initialize() has been called.
    initialized: bool,
    /// Discovered tools from this server.
    tools: Vec<McpToolInfo>,
}

/// High-level MCP client managing one or more server connections.
#[derive(Clone)]
pub struct McpClient {
    /// Default server URL.
    default_url: String,
    /// Active sessions keyed by URL.
    sessions: Arc<RwLock<Vec<ServerSession>>>,
    /// HTTP client — reused across calls for connection pooling.
    http_client: Arc<reqwest::Client>,
}

impl McpClient {
    pub fn new(default_url: impl Into<String>) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| {
                // Fallback to defaults if builder fails
                reqwest::Client::new()
            });
        Self {
            default_url: default_url.into(),
            sessions: Arc::new(RwLock::new(Vec::new())),
            http_client: Arc::new(http_client),
        }
    }

    /// Returns the default server URL.
    pub fn default_url(&self) -> &str {
        &self.default_url
    }

    /// Ensure the URL has a trailing slash.
    fn normalise_url(url: &str) -> String {
        let mut u = url.to_string();
        if !u.ends_with('/') {
            u.push('/');
        }
        u
    }

    /// Initialize a connection to the given URL and discover tools.
    ///
    /// This sends the JSON-RPC `initialize` handshake then `list_tools`.
    pub async fn connect(&self, url: Option<&str>) -> Result<Vec<McpToolInfo>> {
        let target = Self::normalise_url(url.unwrap_or(&self.default_url));

        // 1. Send initialize.
        log::debug!("Sending MCP initialize to {}", target);
        let _init_result = self.jsonrpc_post(
            &target,
            "initialize",
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "agent-harness",
                    "version": "0.1.0",
                }
            }),
        ).await?;

        // 2. Send initialized notification.
        log::debug!("Sending MCP initialized notification to {}", target);
        if let Err(e) = self.jsonrpc_post(
            &target,
            "notifications/initialized",
            Value::Null,
        ).await {
            log::warn!("Failed to send initialized notification to {}: {}", target, e);
        }

        // 3. List tools.
        log::debug!("Requesting tool list from {}", target);
        let list_result = self.jsonrpc_post(&target, "tools/list", Value::Null).await?;

        let tools: Vec<McpToolInfo> = {
            let arr = list_result
                .get("result")
                .and_then(|r| {
                    r.get("tools").cloned().or_else(|| {
                        r.as_array().map(|a| Value::Array(a.clone()))
                    })
                })
                .and_then(|t| t.as_array().map(|a| a.to_vec()));
            arr.map(|a| {
                a.iter().filter_map(|t| {
                    Some(McpToolInfo {
                        name: t.get("name")?.as_str()?.to_string(),
                        description: t.get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .to_string(),
                        input_schema: t.get("inputSchema").cloned(),
                    })
                }).collect()
            })
            .unwrap_or_default()
        };

        // Record session.
        {
            let mut sessions = self.sessions.write().await;
            // Remove stale session for same URL.
            sessions.retain(|s| s.url != target);
            sessions.push(ServerSession {
                url: target.clone(),
                initialized: true,
                tools: tools.clone(),
            });
        }

        log::info!("MCP connection established to '{}' — {} tool(s) discovered", target, tools.len());
        Ok(tools)
    }

    /// Call a tool on the server.
    pub async fn call_tool(
        &self,
        url: Option<&str>,
        tool_name: &str,
        parameters: &Value,
    ) -> Result<Value> {
        let target = Self::normalise_url(url.unwrap_or(&self.default_url));
        log::debug!("Calling MCP tool '{}' on {}", tool_name, target);

        let result = self.jsonrpc_post(
            &target,
            "tools/call",
            serde_json::json!({
                "name": tool_name,
                "arguments": parameters,
            }),
        ).await?;

        log::info!("MCP tool '{}' called successfully on {}", tool_name, target);
        // Extract the result payload.
        Ok(result.get("result")
            .cloned()
            .unwrap_or_else(|| Value::String(format!("Raw: {}", result))))
    }

    /// List all connected server URLs.
    pub async fn connected_servers(&self) -> Vec<String> {
        let sessions = self.sessions.read().await;
        sessions.iter().map(|s| s.url.clone()).collect()
    }

    /// List all discovered tools across all servers.
    pub async fn all_tools(&self) -> Vec<McpToolInfo> {
        let sessions = self.sessions.read().await;
        let mut all = Vec::new();
        for session in sessions.iter() {
            all.extend(session.tools.iter().cloned());
        }
        all
    }

    // ── JSON-RPC transport ──────────────────────────────────────────────────

    async fn jsonrpc_post(&self, url: &str, method: &str, params: Value) -> Result<Value> {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": uuid::Uuid::new_v4().to_string(),
        });

        log::debug!("JSON-RPC → {} {} (url={})", method, payload["id"], url);

        let resp = self.http_client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(&payload)
            .send()
            .await
            .context("POST request failed")?;

        let status = resp.status();
        let body = resp.text().await.context("read response body")?;

        if !status.is_success() {
            let preview = body.chars().take(200).collect::<String>();
            log::error!("HTTP {} from {}: {}", status, url, preview);
            anyhow::bail!("HTTP {} from {}: {}", status, url, preview);
        }

        log::debug!("JSON-RPC response from {} ({} bytes)", url, body.len());
        serde_json::from_str(&body)
            .map_err(|e| {
                log::error!("Failed to parse JSON-RPC response from {}: {}", url, e);
                e
            })
            .context("parse JSON-RPC response")
    }
}

impl Default for McpClient {
    fn default() -> Self {
        Self::new("http://localhost:8463/")
    }
}

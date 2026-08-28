// ToolRegistry — stores, discovers, and executes tools by name.
// Uses interior mutability (Mutex) so tools can be registered from any context
// (e.g., a spawned tokio task) while calls proceed concurrently.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::tool::{Tool, ToolContext, ToolResult};

pub struct ToolRegistry {
	tools: Mutex<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
	pub fn new() -> Self {
		Self {
			tools: Mutex::new(HashMap::new()),
		}
	}

	/// Register a tool. Returns Err if the name is already taken.
	pub fn register(&self, tool: Box<dyn Tool>) -> Result<(), String> {
		let name = tool.name().to_string();
		let mut guard = self.tools.lock().unwrap();
		if guard.contains_key(&name) {
			log::warn!("Tool '{}' already registered — skipping", name);
			return Err(format!("Tool '{}' already registered", name));
		}
		log::info!("Registering tool: '{}'", name);
		guard.insert(name, Arc::from(tool));
		Ok(())
	}

	/// Remove a tool by name.
	pub fn unregister(&self, name: &str) -> Option<Arc<dyn Tool>> {
		let mut guard = self.tools.lock().unwrap();
		let removed = guard.remove(name);
		if removed.is_some() {
			log::info!("Unregistered tool: '{}'", name);
		}
		removed
	}

	/// Call a tool by name.
	///
	/// Clones an Arc reference to the tool so the mutex is held only during
	/// the synchronous lookup. The async call() runs with no locks held.
	pub async fn call(&self, name: &str, ctx: &ToolContext, params: &Value) -> ToolResult {
		let tool: Arc<dyn Tool> = {
			let guard = self.tools.lock().unwrap();
			match guard.get(name) {
				Some(arc) => arc.clone(),
				None => {
					log::warn!("Tool not found: '{}'", name);
					return ToolResult::err(format!("Unknown tool: {}", name));
				}
			}
		};
		// Lock is dropped. Arc keeps the tool alive for the async call.
		log::debug!("Executing tool '{}' with params: {}", name, params);
		let result = tool.call(ctx, params).await;
		if result.success {
			log::info!("Tool '{}' completed successfully", name);
		} else {
			log::warn!("Tool '{}' returned error: {:?}", name, result.error);
		}
		result
	}

	/// List all registered tool names.
	pub fn names(&self) -> Vec<String> {
		let guard = self.tools.lock().unwrap();
		guard.keys().cloned().collect()
	}

	/// Return a summary of every registered tool (name + description) as JSON.
	pub fn to_json(&self) -> Value {
		let guard = self.tools.lock().unwrap();
		let entries: Vec<Value> = guard
			.iter()
			.map(|(name, _tool)| serde_json::json!({ "name": name }))
			.collect();
		serde_json::json!({ "tools": entries })
	}

	pub fn len(&self) -> usize {
		let guard = self.tools.lock().unwrap();
		guard.len()
	}

	pub fn is_empty(&self) -> bool {
		let guard = self.tools.lock().unwrap();
		guard.is_empty()
	}
}

impl Default for ToolRegistry {
	fn default() -> Self {
		Self::new()
	}
}

// Blueprint — a serialized, validated agent design ready for deployment.
//
// A Blueprint is exported from the Anvil and can be saved to JSON/YAML,
// validated, and later re-imported.

use serde::{Deserialize, Serialize};

/// A single slot in a Blueprint (exported from an Anvil slot).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlueprintSlot {
	pub id: String,
	pub ingot_id: String,
	pub properties: std::collections::HashMap<String, serde_json::Value>,
}

/// A link in a Blueprint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlueprintLink {
	pub from: String,
	pub to: String,
	pub label: String,
}

/// A complete Blueprint — ready to be saved, shared, or deployed.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Blueprint {
	pub name: String,
	pub description: String,
	pub version: String,
	pub slots: Vec<BlueprintSlot>,
	pub links: Vec<BlueprintLink>,
	/// Metadata: tags, author, timestamp.
	pub metadata: std::collections::HashMap<String, String>,
}

impl Blueprint {
	pub fn new(name: &str, description: &str) -> Self {
		Self {
			name: name.to_string(),
			description: description.to_string(),
			version: "0.1.0".to_string(),
			slots: Vec::new(),
			links: Vec::new(),
			metadata: std::collections::HashMap::new(),
		}
	}

	/// Validate the blueprint: check for missing references and cycles.
	pub fn validate(&self) -> Vec<String> {
		let mut errors = Vec::new();

		// Check that all link references exist
		let slot_ids: std::collections::HashSet<_> =
			self.slots.iter().map(|s| s.id.as_str()).collect();
		for link in &self.links {
			if !slot_ids.contains(link.from.as_str()) {
				errors.push(format!(
					"Link '{}' references missing slot '{}'",
					link.label, link.from
				));
			}
			if !slot_ids.contains(link.to.as_str()) {
				errors.push(format!(
					"Link '{}' references missing slot '{}'",
					link.label, link.to
				));
			}
		}

		// Check for cycles
		let mut visited = std::collections::HashSet::new();
		let mut rec_stack = std::collections::HashSet::new();
		for slot in &self.slots {
			if !visited.contains(&slot.id) {
				Self::dfs_validate(
					&self.slots,
					&self.links,
					&slot.id,
					&mut visited,
					&mut rec_stack,
					&mut errors,
				);
			}
		}

		errors
	}

	fn dfs_validate(
		slots: &[BlueprintSlot],
		links: &[BlueprintLink],
		node: &str,
		visited: &mut std::collections::HashSet<String>,
		rec_stack: &mut std::collections::HashSet<String>,
		errors: &mut Vec<String>,
	) {
		let node_owned = node.to_string();
		visited.insert(node_owned.clone());
		rec_stack.insert(node_owned.clone());

		for link in links {
			if link.from == node {
				if rec_stack.contains(&link.to) {
					errors.push(format!(
						"Circular dependency detected: {} → {}",
						link.from, link.to
					));
				} else if !visited.contains(&link.to) {
					Self::dfs_validate(slots, links, &link.to, visited, rec_stack, errors);
				}
			}
		}

		rec_stack.remove(node);
	}

	/// Serialize to JSON.
	pub fn to_json(&self) -> Result<String, serde_json::Error> {
		serde_json::to_string_pretty(self)
	}

	/// Deserialize from JSON.
	pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
		serde_json::from_str(json)
	}

	/// Serialize to YAML (requires serde_yaml — falls back to JSON if unavailable).
	pub fn to_yaml(&self) -> String {
		// Simple YAML-like output without external dependency
		let mut yaml = format!(
			"---\nname: {}\ndescription: {}\nversion: {}\nslots:\n",
			self.name, self.description, self.version
		);
		for slot in &self.slots {
			yaml.push_str(&format!(
				"  - id: {}\n    ingot_id: {}\n    properties: {:?}\n",
				slot.id, slot.ingot_id, slot.properties
			));
		}
		yaml.push_str("links:\n");
		for link in &self.links {
			yaml.push_str(&format!(
				"  - from: {}\n    to: {}\n    label: {}\n",
				link.from, link.to, link.label
			));
		}
		yaml
	}

	/// Number of slots.
	pub fn slot_count(&self) -> usize {
		self.slots.len()
	}

	/// Number of links.
	pub fn link_count(&self) -> usize {
		self.links.len()
	}
}

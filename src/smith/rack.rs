// The Rack — holds Ingots (available components) ready to be forged.
//
// An Ingot is a raw component: a tool, prompt block, adapter, or memory module.
// Pick an Ingot from the Rack, place it on the Anvil, and shape it into a Blueprint.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single Ingot — a raw component available for use.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ingot {
	/// Unique identifier.
	pub id: String,
	/// Display name.
	pub name: String,
	/// What it does.
	pub description: String,
	/// Category (e.g. "tool", "prompt", "adapter", "memory").
	pub category: String,
	/// Tags for search.
	pub tags: Vec<String>,
}

impl Ingot {
	pub fn new(id: &str, name: &str, description: &str, category: &str) -> Self {
		Self {
			id: id.to_string(),
			name: name.to_string(),
			description: description.to_string(),
			category: category.to_string(),
			tags: Vec::new(),
		}
	}

	pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
		self.tags = tags.into_iter().map(Into::into).collect();
		self
	}
}

/// The Rack — browse and filter available Ingots.
#[derive(Clone, Debug, Default)]
pub struct Rack {
	ingots: HashMap<String, Ingot>,
	by_category: HashMap<String, Vec<String>>,
}

impl Rack {
	pub fn new() -> Self {
		Self::default()
	}

	/// Place an Ingot on the Rack.
	pub fn add(&mut self, ingot: Ingot) {
		let id = ingot.id.clone();
		let category = ingot.category.clone();
		self.ingots.insert(id.clone(), ingot);
		self.by_category.entry(category).or_default().push(id);
	}

	/// Iterate over all Ingots.
	pub fn iter(&self) -> impl Iterator<Item = &Ingot> {
		self.ingots.values()
	}

	/// Get an Ingot by ID.
	pub fn get(&self, id: &str) -> Option<&Ingot> {
		self.ingots.get(id)
	}

	/// List available categories.
	pub fn categories(&self) -> Vec<&str> {
		self.by_category.keys().map(|s| s.as_str()).collect()
	}

	/// List Ingots in a category.
	pub fn in_category(&self, category: &str) -> Vec<&Ingot> {
		self.by_category
			.get(category)
			.map(|ids| ids.iter().filter_map(|id| self.ingots.get(id)).collect())
			.unwrap_or_default()
	}

	/// Search by keyword.
	pub fn search(&self, query: &str) -> Vec<&Ingot> {
		let q = query.to_lowercase();
		self.ingots
			.values()
			.filter(|i| {
				i.name.to_lowercase().contains(&q)
					|| i.description.to_lowercase().contains(&q)
					|| i.category.to_lowercase().contains(&q)
					|| i.tags.iter().any(|t| t.to_lowercase().contains(&q))
			})
			.collect()
	}

	pub fn len(&self) -> usize {
		self.ingots.len()
	}

	pub fn is_empty(&self) -> bool {
		self.ingots.is_empty()
	}
}

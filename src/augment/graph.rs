use uuid::Uuid;
use serde::{Serialize, Deserialize};


#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Graph {
	pub id: Uuid,
	pub nodes: Vec<Node>,
	pub edges: Vec<Edge>,
}

// ── Graph types ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
	pub id: Uuid,
	pub nodes: [Node; 2],
	pub relationship: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
	pub id: Uuid,
	pub name: String,
	pub edges: Vec<Edge>,
}

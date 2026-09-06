pub mod codebase;
pub mod tasklist;
pub mod tools;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use codebase::CodeRelationship;
use tasklist::TaskRelationship;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum GraphType {
	#[default]
	Task,
	Tool,
	Codebase,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Graph {
	pub id: Uuid,
	pub graph_type: GraphType,
	pub nodes: Vec<Node>,
	pub edges: Vec<Edge>,
}

// ── Graph types ────────────────────────────────────────────────────────────────

impl Graph {
	pub fn new(id: Uuid, graph_type: GraphType, nodes: Vec<Node>, edges: Vec<Edge>) -> Self {
		Self {
			id,
			graph_type,
			nodes,
			edges,
		}
	}
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
	pub id: Uuid,
	pub nodes: [Node; 2],
	pub relationship: EdgeRelationship,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
	pub id: Uuid,
	pub name: String,
	pub data: String,
	pub edges: Vec<Edge>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EdgeRelationship {
	Code(CodeRelationship),
	Task(TaskRelationship),
}

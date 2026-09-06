use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::augment::graphs::{Edge, Node};

/// Relationship type between task nodes in the graph.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TaskRelationship {
	DependsOn,
	Blocks,
	RelatedTo,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Task {
	id: Uuid,
	name: String,
	desc: String,
	due_date: DateTime<Local>,
	subtasks: Vec<Task>,
	rec_date: DateTime<Local>,
	notes: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TaskGraph {
	id: Uuid,
	nodes: Vec<Node>,
	edges: Vec<Edge>,
}

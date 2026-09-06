use serde::{Deserialize, Serialize};

// Re-exported via super::graph for use by consumers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeRelationship {
	Calls,
	DependsOn,
	Contains,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodebaseGraphNodeTypes {
	File,
	Module, // Directory or mod.rs
	Function,
	ObjectType, // Structs, Objects
	UnionType,  // Enums, unions
	ClassType,  // Traits, Classes
}

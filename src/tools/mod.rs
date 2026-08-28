// Tool system — trait, result, error, registry, and built-in implementations.

pub mod filesystem;
pub mod registry;
pub mod sandbox;
pub mod tool;
pub mod web;

pub use registry::ToolRegistry;
pub use tool::{Tool, ToolContext, ToolError, ToolResult};

// Tool system — trait, result, error, registry, and built-in implementations.

pub mod tool;
pub mod registry;
pub mod sandbox;
pub mod filesystem;
pub mod web;

pub use tool::{Tool, ToolContext, ToolResult, ToolError};
pub use registry::ToolRegistry;

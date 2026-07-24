// Imports needed: serde::Serialize/Deserialize for event serialization, uuid::Uuid for unique IDs
// This module defines all possible events that flow through the application

/// Enum representing all possible events in the application
/// Used by event_bus for publishing, by all modules for subscribing and handling
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AppEvent {
    /// Keyboard input event from crossterm
    /// Contains KeyCode and KeyModifiers for input processing
    KeyboardInput(crossterm::event::KeyEvent),
    
    /// Mouse input event from crossterm
    /// Contains MouseEventKind and position for UI interaction
    MouseInput(crossterm::event::MouseEvent),
    
    /// Terminal resize event
    /// Contains new width and height for layout recalculation
    Resize(u16, u16),
    
    /// Chat message from user
    /// Contains message text and optional conversation ID
    ChatMessage { content: String, conversation_id: Option<uuid::Uuid> },
    
    /// Tool execution request
    /// Contains tool name and serialized parameters
    ToolRequest { tool_name: String, params: serde_json::Value },
    
    /// Tool execution result
    /// Contains tool name, success status, and result or error
    ToolResult { tool_name: String, success: bool, result: String },
    
    /// Agent planning cycle start
    /// Contains planning agent ID and context
    PlanningStart { agent_id: uuid::Uuid, context: String },
    
    /// Agent generation cycle start
    /// Contains generating agent ID and task list
    GenerationStart { agent_id: uuid::Uuid, tasks: Vec<String> },
    
    /// Agent evaluation cycle complete
    /// Contains evaluator agent ID and completion status map
    EvaluationComplete { agent_id: uuid::Uuid, completed: std::collections::HashMap<String, bool> },
    
    /// Autoresearch iteration start
    /// Contains metric name, target value, and iteration number
    AutoresearchStart { metric: String, target: f64, iteration: u32 },
    
    /// Autoresearch iteration result
    /// Contains metric improvement status and branch name
    AutoresearchResult { improved: bool, branch_name: String, metric_value: f64 },
    
    /// MCP connection established
    /// Contains server URL and available tools count
    McpConnected { server_url: String, tools_available: usize },
    
    /// MCP connection lost
    /// Contains server URL and error message
    McpDisconnected { server_url: String, reason: String },
    
    /// Component loaded in Agent Smith
    /// Contains component type and name
    ComponentLoaded { component_type: String, name: String },
    
    /// Component assembled into agent design
    /// Contains agent design ID and list of component names
    ComponentAssembled { design_id: uuid::Uuid, components: Vec<String> },
    
    /// Memory allocation request
    /// Contains size in bytes and memory type (GPU/CPU)
    MemoryAllocate { size: usize, memory_type: MemoryType },
    
    /// Memory deallocation notification
    /// Contains memory handle ID
    MemoryDeallocate { handle_id: u64 },
    
    /// Render trigger for TUI update
    /// Forces immediate redraw of terminal
    Render,
    
    /// Shutdown signal for graceful termination
    Shutdown,
}

/// Enum for memory type specification in memory events
/// Used by memory_controller for GPU vs CPU allocation decisions
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MemoryType {
    /// System RAM - slower but larger capacity
    Cpu,
    /// GPU VRAM - faster for parallel operations
    Gpu,
    /// Unified memory pool managed by software controller
    Unified,
}

/// Helper struct for keyboard event abstraction
/// Used by tui_shell to normalize crossterm key events
pub struct KeyboardEvents {
    /// Quit command key binding
    pub code_q: crossterm::event::KeyCode,
    /// Refresh/redraw command key binding
    pub code_r: crossterm::event::KeyCode,
    /// Input mode toggle key binding
    pub code_i: crossterm::event::KeyCode,
    /// Tools panel toggle key binding
    pub code_t: crossterm::event::KeyCode,
}

/// Implements KeyboardEvents with default key bindings
impl KeyboardEvents {
    /// Creates KeyboardEvents with default Q, R, I, T bindings
    /// Returns KeyboardEvents struct
    pub fn new() -> Self;
    
    /// Checks if a KeyEvent matches any registered binding
    /// Takes &self and KeyEvent, returns Option<&'static str> with action name
    pub fn match_key(&self, key: crossterm::event::KeyEvent) -> Option<&'static str>;
}

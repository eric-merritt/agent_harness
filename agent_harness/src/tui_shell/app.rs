// Imports needed: tokio for async runtime, SharedEventBus from event_core, ComponentRegistry from component module
// This module defines the main application struct and entry point for the TUI render loop

/// Main application struct holding all state and managing the render loop
/// Used by main.rs as entry point, coordinates between all UI components
pub struct App {
    // Shared event bus for inter-module communication
    event_bus: SharedEventBus,
    // Registry of all active UI components
    component_registry: ComponentRegistry,
    // Current screen layout configuration
    layout: Layout,
    // Flag indicating if app should continue running
    running: bool,
    // Flag indicating if render is needed
    pending_render: bool,
}

/// Implements App with main run loop and lifecycle management
impl App {
    /// Creates new App instance with initialized event bus and components
    /// Returns Result<App, Error> - fails if terminal setup fails
    pub fn new() -> Result<Self, Box<dyn std::error::Error>>;

    /// Main application run loop - captures input, processes events, renders UI
    /// Takes &mut self, returns Result<(), Error> on shutdown
    /// Loop structure: poll events -> dispatch to handlers -> render if dirty -> sleep
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>>;

    /// Handles a single event from the event bus
    /// Takes &mut self and &AppEvent, returns Result<bool, Error> indicating consumption
    async fn handle_event(&mut self, event: &AppEvent) -> Result<bool, Box<dyn std::error::Error>>;

    /// Triggers a full screen redraw
    /// Takes &mut self, no return value
    fn request_render(&mut self);

    /// Gracefully shuts down the application
    /// Takes &mut self, returns Result<(), Error> after cleanup
    pub fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}

/// Application state enum for tracking current mode
/// Used by App to determine which components are active
#[derive(Clone, Debug, PartialEq)]
pub enum AppState {
    /// Normal chat interaction mode
    Chat,
    /// Tool selection and execution mode
    Tools,
    /// Agent Smith component assembly mode
    AgentSmith,
    /// Autoresearch loop monitoring mode
    Autoresearch,
}

/// Configuration struct for terminal settings
/// Used during App initialization for crossterm setup
pub struct TerminalConfig {
    /// Enable mouse capture for clickable UI elements
    pub mouse_enabled: bool,
    /// Enable raw mode for direct key capture
    pub raw_mode: bool,
    /// Enable alternate screen buffer
    pub alt_screen: bool,
    /// Frame rate limit for rendering (fps)
    pub fps_limit: u32,
}

/// Implements TerminalConfig with sensible defaults
impl TerminalConfig {
    /// Creates TerminalConfig with default settings
    /// Returns TerminalConfig with mouse enabled, 60 fps limit
    pub fn default() -> Self;
}

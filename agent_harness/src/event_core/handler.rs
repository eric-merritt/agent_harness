// Imports needed: async_trait for trait macros, AppEvent from events module
// This module defines the trait that all modules must implement to receive events

/// Trait for async event handling - implemented by all modular components
/// Used by event_core to dispatch events, by each module (tui_shell, tool_engine, agent_loops, etc.)
#[async_trait::async_trait]
pub trait EventHandler: Send + Sync {
    /// Returns the unique name identifier for this handler
    /// Takes &self, returns &'static str used for logging and debugging
    fn name(&self) -> &'static str;

    /// Processes an incoming event asynchronously
    /// Takes &self and AppEvent reference, returns Result<bool, Error>
    /// Returns Ok(true) if event was consumed, Ok(false) if should propagate
    async fn handle_event(&self, event: &AppEvent) -> Result<bool, Box<dyn std::error::Error + Send>>;

    /// Called when handler is registered with the event bus
    /// Takes &self and SharedEventBus reference, returns Result<(), Error>
    /// Used for initialization and subscription setup
    async fn on_register(&self, event_bus: &SharedEventBus) -> Result<(), Box<dyn std::error::Error + Send>>;

    /// Called when handler is unregistered from the event bus
    /// Takes &self, returns Result<(), Error> for cleanup operations
    async fn on_unregister(&self) -> Result<(), Box<dyn std::error::Error + Send>>;

    /// Returns priority level for event processing order
    /// Takes &self, returns u8 where lower numbers = higher priority
    /// High priority handlers (0-10) process before low priority (100+)
    fn priority(&self) -> u8 {
        50
    }
}

/// Container for registered event handlers with priority ordering
/// Used internally by event_core to manage handler lifecycle
pub struct HandlerRegistry {
    // Vector of boxed trait objects sorted by priority
    handlers: Vec<Box<dyn EventHandler>>,
    // DashMap for O(1) lookup by handler name
    handler_map: dashmap::DashMap<String, usize>,
}

/// Implements HandlerRegistry for adding, removing, and dispatching to handlers
impl HandlerRegistry {
    /// Creates empty HandlerRegistry
    /// Returns HandlerRegistry with capacity pre-allocated
    pub fn new() -> Self;

    /// Registers a new handler maintaining priority order
    /// Takes &mut self and Box<dyn EventHandler>, returns Result<usize, Error> with index
    pub fn register(&mut self, handler: Box<dyn EventHandler>) -> Result<usize, Box<dyn std::error::Error + Send>>;

    /// Unregisters handler by name
    /// Takes &mut self and &str name, returns Option<Box<dyn EventHandler>>
    pub fn unregister(&mut self, name: &str) -> Option<Box<dyn EventHandler>>;

    /// Dispatches event to all handlers in priority order
    /// Takes &self and &AppEvent, returns Vec<Result<bool, Error>> with each handler's result
    pub async fn dispatch(&self, event: &AppEvent) -> Vec<Result<bool, Box<dyn std::error::Error + Send>>>;

    /// Returns count of registered handlers
    /// Takes &self, returns usize count
    pub fn count(&self) -> usize;
}

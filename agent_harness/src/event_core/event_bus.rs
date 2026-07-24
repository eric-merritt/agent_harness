// Imports needed: tokio::sync::broadcast for multi-subscriber channel, serde::Serialize for event serialization
// This module defines the central event bus that all modules use for communication

/// Struct to hold the broadcast sender for publishing events to all subscribers
/// Uses tokio broadcast channel for efficient multi-subscriber async messaging
pub struct EventBus {
    // Sender side of broadcast channel - cloned for each publisher
    sender: tokio::sync::broadcast::Sender<AppEvent>,
    // Internal counter for tracking active subscribers
    subscriber_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

/// Implements EventBus with methods for publish and subscribe operations
impl EventBus {
    /// Creates a new EventBus with specified buffer capacity
    /// Returns Result<EventBus, Error> - fails if capacity is zero
    pub fn new(capacity: usize) -> Self;

    /// Publishes an event to all subscribed handlers
    /// Takes &self and AppEvent, returns Result<usize, SendError> indicating subscriber count
    pub fn publish(&self, event: AppEvent) -> Result<usize, tokio::sync::broadcast::error::SendError<AppEvent>>;

    /// Subscribes to receive events from the bus
    /// Takes &self, returns broadcast::Receiver<AppEvent> for async event consumption
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<AppEvent>;

    /// Returns current number of active subscribers
    /// Takes &self, returns usize count of subscribers
    pub fn subscriber_count(&self) -> usize;
}

/// Thread-safe wrapper for sharing EventBus across async tasks
/// Uses Arc for reference counting without mutex overhead
pub struct SharedEventBus {
    inner: std::sync::Arc<EventBus>,
}

/// Implements SharedEventBus for cloning and thread-safe access
impl SharedEventBus {
    /// Wraps an EventBus in Arc for shared ownership
    /// Takes EventBus, returns SharedEventBus
    pub fn new(bus: EventBus) -> Self;

    /// Clones the Arc reference - cheap operation
    /// Takes &self, returns SharedEventBus
    pub fn clone(&self) -> Self;

    /// Accesses inner EventBus for publish/subscribe operations
    /// Takes &self, returns &EventBus reference
    pub fn inner(&self) -> &EventBus;
}

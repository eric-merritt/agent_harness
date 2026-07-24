// Module declaration for event_core - the central event bus system
// This module handles all inter-module communication via publish-subscribe pattern

pub mod event_bus;
// Defines EventBus struct and publish/subscribe methods for async message passing

pub mod events;
// Defines AppEvent enum with all possible event types across the application

pub mod handler;
// Defines EventHandler trait for modules to implement async event processing

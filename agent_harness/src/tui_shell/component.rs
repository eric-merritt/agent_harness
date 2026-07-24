// Imports needed: async_trait for trait macros, ratatui::Frame for rendering, AppEvent from event_core
// This module defines the Component trait for UI widgets and registry for management

/// Trait for UI components - implemented by all interactive widgets
/// Used by component_registry to manage lifecycle, by chat_interface, toolbar, etc.
#[async_trait::async_trait]
pub trait Component: Send + Sync {
    /// Returns unique identifier for this component
    /// Takes &self, returns &str used for registry lookup
    fn id(&self) -> &str;

    /// Renders the component to the given frame area
    /// Takes &self, &mut Frame, and Rect area, no return
    /// Draws component content within the specified rectangle
    fn render(&self, frame: &mut ratatui::Frame, area: ratatui::layout::Rect);

    /// Handles an event directed at this component
    /// Takes &mut self and &AppEvent, returns bool indicating consumption
    /// Returns true if event handled, false to propagate to other components
    fn handle_event(&mut self, event: &AppEvent) -> bool;

    /// Called when component gains focus
    /// Takes &mut self, no return - used for state initialization
    fn on_focus(&mut self);

    /// Called when component loses focus
    /// Takes &mut self, no return - used for cleanup or state save
    fn on_blur(&mut self);

    /// Returns minimum size required for proper rendering
    /// Takes &self, returns (u16, u16) width and height
    fn min_size(&self) -> (u16, u16);

    /// Returns preferred size for optimal rendering
    /// Takes &self, returns Option<(u16, u16)> with preferred dimensions
    fn preferred_size(&self) -> Option<(u16, u16)>;
}

/// Registry for managing all active UI components
/// Used by App to store, retrieve, and iterate over components
pub struct ComponentRegistry {
    // Map of component ID to boxed trait object
    components: std::collections::HashMap<String, Box<dyn Component>>,
    // Ordered list of component IDs for render order
    render_order: Vec<String>,
    // Currently focused component ID
    focused_component: Option<String>,
}

/// Implements ComponentRegistry with CRUD operations for components
impl ComponentRegistry {
    /// Creates empty ComponentRegistry
    /// Returns ComponentRegistry with pre-allocated capacity
    pub fn new() -> Self;

    /// Registers a component in the registry
    /// Takes &mut self and Box<dyn Component>, returns Result<(), Error> if ID exists
    pub fn register(&mut self, component: Box<dyn Component>) -> Result<(), Box<dyn std::error::Error>>;

    /// Unregisters a component by ID
    /// Takes &mut self and &str id, returns Option<Box<dyn Component>>
    pub fn unregister(&mut self, id: &str) -> Option<Box<dyn Component>>;

    /// Gets mutable reference to component by ID
    /// Takes &mut self and &str id, returns Option<&mut dyn Component>
    pub fn get_mut(&mut self, id: &str) -> Option<&mut dyn Component>;

    /// Gets immutable reference to component by ID
    /// Takes &self and &str id, returns Option<&dyn Component>
    pub fn get(&self, id: &str) -> Option<&dyn Component>;

    /// Sets which component has focus
    /// Takes &mut self and &str id, returns bool indicating success
    pub fn set_focus(&mut self, id: &str) -> bool;

    /// Returns ID of currently focused component
    /// Takes &self, returns Option<&str>
    pub fn focused(&self) -> Option<&str>;

    /// Dispatches event to focused component first, then others in order
    /// Takes &mut self and &AppEvent, returns bool indicating if any component handled it
    pub fn dispatch_event(&mut self, event: &AppEvent) -> bool;

    /// Renders all components in registered order
    /// Takes &self, &mut Frame, and full screen Rect
    pub fn render_all(&self, frame: &mut ratatui::Frame, area: ratatui::layout::Rect);

    /// Returns count of registered components
    /// Takes &self, returns usize count
    pub fn len(&self) -> usize;

    /// Returns true if registry is empty
    /// Takes &self, returns bool
    pub fn is_empty(&self) -> bool;
}

// Imports needed: ratatui::layout::Rect for area definitions, serde for serialization
// This module defines the layout system for composing screen regions

/// Struct representing a named screen region with position and size
/// Used by Layout to define renderable areas, by components to claim space
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Zone {
    /// Unique identifier for this zone
    pub id: String,
    /// X coordinate of top-left corner
    pub x: u16,
    /// Y coordinate of top-left corner
    pub y: u16,
    /// Width of the zone in terminal columns
    pub width: u16,
    /// Height of the zone in terminal rows
    pub height: u16,
    /// Priority for resize operations - higher priority zones keep size
    pub priority: u8,
}

/// Implements Zone with construction and manipulation methods
impl Zone {
    /// Creates a new Zone with specified position and dimensions
    /// Takes id: String, x, y, width, height: u16, returns Zone
    pub fn new(id: String, x: u16, y: u16, width: u16, height: u16) -> Self;

    /// Converts Zone to ratatui Rect for rendering
    /// Takes &self, returns Rect with same coordinates
    pub fn to_rect(&self) -> ratatui::layout::Rect;

    /// Resizes zone to new dimensions while maintaining position
    /// Takes &mut self, width: u16, height: u16, no return
    pub fn resize(&mut self, width: u16, height: u16);

    /// Moves zone to new position without changing size
    /// Takes &mut self, x: u16, y: u16, no return
    pub fn move_to(&mut self, x: u16, y: u16);

    /// Checks if a point is within this zone's bounds
    /// Takes &self, x: u16, y: u16, returns bool
    pub fn contains(&self, x: u16, y: u16) -> bool;
}

/// Enum defining layout composition strategies
/// Used by App to determine how zones are arranged on screen
#[derive(Clone, Debug, PartialEq)]
pub enum LayoutType {
    /// Single full-screen zone
    Fullscreen,
    /// Horizontal split between zones
    HorizontalSplit,
    /// Vertical split between zones
    VerticalSplit,
    /// Grid layout with rows and columns
    Grid { rows: usize, cols: usize },
    /// Overlay mode with z-indexed zones
    Overlay,
}

/// Struct holding complete layout configuration
/// Used by App to manage all zones and calculate positions
pub struct Layout {
    /// All zones in this layout
    pub zones: Vec<Zone>,
    /// Layout type determining arrangement strategy
    pub layout_type: LayoutType,
    /// Terminal width for responsive calculations
    pub terminal_width: u16,
    /// Terminal height for responsive calculations
    pub terminal_height: u16,
}

/// Implements Layout with zone management and calculation methods
impl Layout {
    /// Creates new Layout with default configuration
    /// Returns Layout with fullscreen zone and empty zone list
    pub fn new() -> Self;

    /// Adds a zone to the layout
    /// Takes &mut self and Zone, returns &mut Zone reference
    pub fn add_zone(&mut self, zone: Zone) -> &mut Zone;

    /// Removes a zone by ID
    /// Takes &mut self and &str id, returns Option<Zone>
    pub fn remove_zone(&mut self, id: &str) -> Option<Zone>;

    /// Gets zone by ID
    /// Takes &self and &str id, returns Option<&Zone>
    pub fn get_zone(&self, id: &str) -> Option<&Zone>;

    /// Recalculates all zone positions based on layout type and terminal size
    /// Takes &mut self, no return - mutates zone coordinates
    pub fn recalculate(&mut self);

    /// Finds zone containing a point
    /// Takes &self, x: u16, y: u16, returns Option<&Zone>
    pub fn zone_at(&self, x: u16, y: u16) -> Option<&Zone>;

    /// Handles terminal resize event
    /// Takes &mut self, new_width: u16, new_height: u16, recalculates layout
    pub fn on_resize(&mut self, new_width: u16, new_height: u16);
}

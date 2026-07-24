// Imports needed: ratatui::Frame, ratatui::layout::Rect, AppEvent from event_core, Component trait from component module
// This module implements the toolbar component with clickable tabs for tool selection

/// Struct representing toolbar with tabbed interface for tool categories
/// Implements Component trait, used for horizontal tool selection bar
pub struct Toolbar {
    // Unique component identifier
    id: String,
    // Vector of toolbar tabs with labels and icons
    tabs: Vec<ToolbarTab>,
    // Index of currently selected tab
    active_tab: usize,
    // Screen coordinates for hit testing clicks
    tab_rects: Vec<ratatui::layout::Rect>,
    // Flag indicating if mouse hover effects are enabled
    hover_enabled: bool,
    // Currently hovered tab index
    hovered_tab: Option<usize>,
}

/// Implements Toolbar with tab management and rendering
impl Toolbar {
    /// Creates new Toolbar with default tool category tabs
    /// Returns Toolbar with predefined tabs for Tools, Agents, Settings
    pub fn new() -> Self;

    /// Adds a new tab to the toolbar
    /// Takes &mut self, label: String, icon: char, returns usize index
    pub fn add_tab(&mut self, label: String, icon: char) -> usize;

    /// Removes a tab by index
    /// Takes &mut self and usize index, returns Option<ToolbarTab>
    pub fn remove_tab(&mut self, index: usize) -> Option<ToolbarTab>;

    /// Sets the active tab by index
    /// Takes &mut self and usize index, no return
    pub fn set_active_tab(&mut self, index: usize);

    /// Gets the currently active tab index
    /// Takes &self, returns usize index
    pub fn active_tab(&self) -> usize;

    /// Gets tab by index
    /// Takes &self and usize index, returns Option<&ToolbarTab>
    pub fn get_tab(&self, index: usize) -> Option<&ToolbarTab>;

    /// Returns number of tabs in toolbar
    /// Takes &self, returns usize count
    pub fn tab_count(&self) -> usize;

    /// Finds tab index by label
    /// Takes &self and &str label, returns Option<usize>
    pub fn find_tab_by_label(&self, label: &str) -> Option<usize>;
}

/// Implements Component trait for Toolbar
impl Component for Toolbar {
    /// Returns component ID "toolbar"
    fn id(&self) -> &str;

    /// Renders toolbar tabs with highlighting for active/hovered state
    /// Takes &self, &mut Frame, Rect area - draws horizontal tab bar
    fn render(&self, frame: &mut ratatui::Frame, area: ratatui::layout::Rect);

    /// Handles mouse clicks and keyboard navigation for tab selection
    /// Takes &mut self and &AppEvent, returns bool indicating consumption
    fn handle_event(&mut self, event: &AppEvent) -> bool;

    /// Called when toolbar gains focus
    fn on_focus(&mut self);

    /// Called when toolbar loses focus
    fn on_blur(&mut self);

    /// Returns minimum size (full width x 3 rows)
    fn min_size(&self) -> (u16, u16);

    /// Returns preferred size (full width x 5 rows)
    fn preferred_size(&self) -> Option<(u16, u16)>;
}

/// Struct representing a single toolbar tab
/// Used by Toolbar to store tab configuration
#[derive(Clone, Debug)]
pub struct ToolbarTab {
    /// Display label for the tab
    pub label: String,
    /// Optional icon character for visual identification
    pub icon: Option<char>,
    /// Associated tool category or action ID
    pub action_id: String,
    /// Flag indicating if tab has unread notifications
    pub has_notification: bool,
    /// Notification count badge value
    pub notification_count: Option<u32>,
}

/// Implements ToolbarTab with construction helper
impl ToolbarTab {
    /// Creates new ToolbarTab with label and optional icon
    /// Takes label: String, icon: Option<char>, action_id: String, returns ToolbarTab
    pub fn new(label: String, icon: Option<char>, action_id: String) -> Self;
}

/// Enum for toolbar orientation
/// Used by Toolbar to determine layout direction
#[derive(Clone, Debug, PartialEq)]
pub enum ToolbarOrientation {
    /// Horizontal bar at top or bottom
    Horizontal,
    /// Vertical bar at left or right
    Vertical,
}

/// Enum for tab position in toolbar
/// Used during rendering for border styling
#[derive(Clone, Debug, PartialEq)]
pub enum TabPosition {
    /// First tab in toolbar
    First,
    /// Middle tab with neighbors on both sides
    Middle,
    /// Last tab in toolbar
    Last,
    /// Single standalone tab
    Single,
}

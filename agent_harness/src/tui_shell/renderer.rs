// Imports needed: ratatui::Terminal, crossterm::terminal for terminal management, ratatui::Frame for rendering
// This module wraps ratatui Terminal with application-specific render methods

/// Struct wrapping ratatui Terminal with buffer management
/// Used by App for all terminal output, handles draw calls and flush
pub struct Renderer {
    // Underlying ratatui Terminal instance
    terminal: ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    // Flag indicating if terminal is in alternate screen mode
    alt_screen_active: bool,
    // Frame counter for performance monitoring
    frame_count: u64,
    // Timestamp of last frame for FPS calculation
    last_frame_time: std::time::Instant,
}

/// Implements Renderer with terminal lifecycle and rendering methods
impl Renderer {
    /// Creates new Renderer with initialized terminal
    /// Returns Result<Renderer, Error> - fails if terminal setup fails
    pub fn new() -> Result<Self, Box<dyn std::error::Error>>;

    /// Renders a frame using the provided closure
    /// Takes &mut self and closure Fn(&mut Frame), returns Result<(), Error>
    /// Closure receives Frame for drawing widgets to specified areas
    pub fn render<F>(&mut self, f: F) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnOnce(&mut ratatui::Frame);

    /// Clears the terminal screen completely
    /// Takes &mut self, returns Result<(), Error>
    pub fn clear(&mut self) -> Result<(), Box<dyn std::error::Error>>;

    /// Enters alternate screen mode for fullscreen app
    /// Takes &mut self, returns Result<(), Error>
    pub fn enter_alt_screen(&mut self) -> Result<(), Box<dyn std::error::Error>>;

    /// Exits alternate screen mode returning to normal terminal
    /// Takes &mut self, returns Result<(), Error>
    pub fn exit_alt_screen(&mut self) -> Result<(), Box<dyn std::error::Error>>;

    /// Enables mouse capture for click events
    /// Takes &mut self, returns Result<(), Error>
    pub fn enable_mouse(&mut self) -> Result<(), Box<dyn std::error::Error>>;

    /// Disables mouse capture
    /// Takes &mut self, returns Result<(), Error>
    pub fn disable_mouse(&mut self) -> Result<(), Box<dyn std::error::Error>>;

    /// Returns current frames per second
    /// Takes &self, returns f64 calculated from frame timing
    pub fn fps(&self) -> f64;

    /// Returns total frame count since initialization
    /// Takes &self, returns u64 count
    pub fn frame_count(&self) -> u64;
}

/// Drop implementation for cleanup on Renderer destruction
/// Ensures terminal is restored to normal state even on panic
impl Drop for Renderer {
    /// Cleans up terminal state on drop
    /// Automatically called when Renderer goes out of scope
    fn drop(&mut self);
}

/// Configuration for render options
/// Used during Renderer initialization
pub struct RenderConfig {
    /// Enable vsync-like frame limiting
    pub limit_fps: bool,
    /// Target FPS when limiting enabled
    pub target_fps: u32,
    /// Enable cursor visibility
    pub cursor_visible: bool,
    /// Cursor style (block, line, underscore)
    pub cursor_style: ratatui::cursor::CursorStyle,
}

/// Implements RenderConfig with defaults
impl RenderConfig {
    /// Creates RenderConfig with default settings
    /// Returns RenderConfig with 60 fps limit, hidden cursor
    pub fn default() -> Self;
}

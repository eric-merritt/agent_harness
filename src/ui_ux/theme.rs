// Theme module — centralizes all color/style tokens used by UI components.

use ratatui::style::{Color, Modifier, Style};

/// Named preset identifier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThemeName {
    #[default]
    Dark,
    Light,
    Retro,
}

/// A complete set of UI color tokens.
#[derive(Clone, Debug)]
pub struct Theme {
    pub name: String,
    pub bg: Color,
    pub fg: Color,
    pub primary: Color,
    pub secondary: Color,
    pub active: Color,
    pub hover: Color,
    pub border: Color,
    pub error: Color,
    pub success: Color,
    pub warning: Color,
    pub reversed_bg: Color,
    pub reversed_fg: Color,
    pub user_message_bg: Color,
    pub message_text_fg: Color,
    pub panel_bg: Color,
    pub title_fg: Color,
    pub cursor_fg: Color,
    pub button_fg: Color,
    pub button_bg: Color,
}

// ── Pre-built presets ──────────────────────────────────────────────────────────

impl Theme {
    /// Dark terminal (black background, light text).
    pub fn dark() -> Self {
        Self {
            name: "dark".to_string(),
            bg: Color::Black,
            fg: Color::White,
            primary: Color::Cyan,
            secondary: Color::Gray,
            active: Color::Yellow,
            hover: Color::Green,
            border: Color::Gray,
            error: Color::Red,
            success: Color::Green,
            warning: Color::Yellow,
            reversed_bg: Color::Yellow,
            reversed_fg: Color::Black,
            user_message_bg: Color::Rgb(112, 128, 144),
            message_text_fg: Color::White,
            panel_bg: Color::Rgb(20, 20, 30),
            title_fg: Color::Rgb(140, 160, 200),
            cursor_fg: Color::Rgb(200, 220, 200),
            button_fg: Color::Rgb(220, 220, 80),
            button_bg: Color::Rgb(40, 40, 60),
        }
    }

    /// Light terminal (white background, dark text).
    pub fn light() -> Self {
        Self {
            name: "light".to_string(),
            bg: Color::White,
            fg: Color::Black,
            primary: Color::Blue,
            secondary: Color::DarkGray,
            active: Color::Magenta,
            hover: Color::Cyan,
            border: Color::DarkGray,
            error: Color::Red,
            success: Color::Green,
            warning: Color::Yellow,
            reversed_bg: Color::DarkGray,
            reversed_fg: Color::White,
            user_message_bg: Color::Rgb(170, 180, 190),
            message_text_fg: Color::White,
            panel_bg: Color::Rgb(240, 240, 245),
            title_fg: Color::Rgb(60, 60, 80),
            cursor_fg: Color::Rgb(40, 40, 40),
            button_fg: Color::Rgb(40, 40, 120),
            button_bg: Color::Rgb(200, 200, 220),
        }
    }

    /// Retro terminal — light gray outside, dark magenta/teal panels.
    pub fn retro() -> Self {
        Self {
            name: "retro".to_string(),
            bg: Color::Rgb(192, 192, 192),       // light gray outer
            fg: Color::Rgb(30, 30, 30),           // dark text on gray
            primary: Color::Rgb(0, 200, 180),     // teal accent
            secondary: Color::Rgb(180, 50, 180),  // magenta accent
            active: Color::Rgb(0, 255, 200),      // bright teal
            hover: Color::Rgb(220, 80, 220),      // bright magenta
            border: Color::Rgb(0, 170, 150),      // teal border
            error: Color::Rgb(220, 50, 50),
            success: Color::Rgb(0, 200, 100),
            warning: Color::Rgb(220, 180, 0),
            reversed_bg: Color::Rgb(100, 20, 100),// dark magenta
            reversed_fg: Color::Rgb(0, 220, 180), // teal on magenta
            user_message_bg: Color::Rgb(120, 30, 120),
            message_text_fg: Color::Rgb(0, 220, 180),
            panel_bg: Color::Rgb(80, 10, 80),     // dark magenta panel bg
            title_fg: Color::Rgb(0, 220, 180),    // teal title
            cursor_fg: Color::Rgb(0, 255, 200),   // bright teal cursor
            button_fg: Color::Rgb(0, 220, 180),   // teal button text
            button_bg: Color::Rgb(100, 20, 100),  // dark magenta button
        }
    }

    /// Build a theme from a named preset.
    pub fn from_name(name: ThemeName) -> Self {
        match name {
            ThemeName::Dark => Self::dark(),
            ThemeName::Light => Self::light(),
            ThemeName::Retro => Self::retro(),
        }
    }
}

// ── Convenience helpers ─────────────────────────────────────────────────────────

impl Theme {
    pub fn active_tab_style(&self) -> Style {
        Style::default()
            .fg(self.active)
            .bg(self.reversed_bg)
            .add_modifier(Modifier::REVERSED)
    }

    pub fn hover_tab_style(&self) -> Style {
        Style::default().fg(self.hover)
    }

    pub fn default_tab_style(&self) -> Style {
        Style::default().fg(self.fg).bg(self.bg)
    }

    pub fn border_style(&self) -> Style {
        Style::default().fg(self.border)
    }

    pub fn error_style(&self) -> Style {
        Style::default().fg(self.error)
    }

    pub fn success_style(&self) -> Style {
        Style::default().fg(self.success)
    }

    pub fn warning_style(&self) -> Style {
        Style::default().fg(self.warning)
    }

    pub fn primary_style(&self) -> Style {
        Style::default().fg(self.primary)
    }

    pub fn user_message_style(&self) -> Style {
        Style::default()
            .fg(self.message_text_fg)
            .bg(self.user_message_bg)
    }

    pub fn message_text_style(&self) -> Style {
        Style::default().fg(self.message_text_fg)
    }
}

/// Thread-safe shared theme.
pub type SharedTheme = std::sync::Arc<tokio::sync::RwLock<Theme>>;

pub fn default_shared_theme() -> SharedTheme {
    std::sync::Arc::new(tokio::sync::RwLock::new(Theme::retro()))
}

pub fn shared_theme_from(name: ThemeName) -> SharedTheme {
    std::sync::Arc::new(tokio::sync::RwLock::new(Theme::from_name(name)))
}

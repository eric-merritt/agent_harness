// Module declaration for tui_shell - the terminal user interface layer
// This module handles all rendering, input capture, and UI component management

pub mod app;
// Defines main App struct with render loop and state management

pub mod layout;
// Defines Zone struct and Layout enum for screen area composition

pub mod renderer;
// Defines Renderer struct wrapping ratatui Terminal with draw methods

pub mod component;
// Defines Component trait for UI widgets and ComponentRegistry for management

pub mod chat_interface;
// Defines ChatInterface component implementing main conversation view

pub mod toolbar;
// Defines Toolbar component with clickable tabs for tool selection

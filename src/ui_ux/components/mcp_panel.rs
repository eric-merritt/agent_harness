// MCP panel — card-style blocks with collapsible detail dropdowns.

use std::sync::{Arc, RwLock};

use ratatui::Frame;
use ratatui::layout::{Rect, Size};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Text;
use ratatui::widgets::{Block, Paragraph};
use tui_scrollview::{ScrollView, ScrollViewState, ScrollbarVisibility};

/// Near-black used as component background so panels pop off the gray terminal.
const PANEL_BG: Color = Color::Rgb(16, 16, 16);
/// Card background — slightly lighter so cards stand out from the panel.
/// Light text on the dark panel.
const TEXT_FG: Color = Color::Rgb(220, 220, 220);
/// Border color — neutral gray.
const BORDER_FG: Color = Color::Rgb(120, 120, 120);
/// Header background — teal (user role color), since this panel sits next to a magenta button.
const HEADER_BG: Color = Color::Rgb(0, 128, 128);
/// Header text — white.
const HEADER_FG: Color = Color::Rgb(255, 255, 255);
/// Dim text for secondary info.
const DIM_FG: Color = Color::Rgb(160, 160, 160);
/// Connected status.
const CONNECTED_FG: Color = Color::Rgb(0, 200, 100);
/// Disconnected status.
const DISCONNECTED_FG: Color = Color::Rgb(220, 180, 0);

/// Box-drawing characters for borders
const LEFT: char = '\u{258E}';
const BOTTOM: char = '\u{2581}';
const RIGHT: char = '\u{258A}';
const CORNER_BL: char = '\u{259D}';
const CORNER_BR: char = '\u{2598}';

/// Represents an MCP server card with optional expanded details.
#[derive(Clone, Debug, Default)]
pub struct McpCard {
    pub name: String,
    pub transport: String,
    pub endpoint: String,
    pub connected: bool,
    pub tools_count: usize,
    pub expanded: bool,
}

pub struct McpPanel {
    pub cards: Vec<McpCard>,
    pub add_button_rect: Option<Rect>,
    pub add_button_pressed: bool,
    pub card_rects: Vec<Rect>,
    pub tool_tree: std::sync::Arc<std::sync::RwLock<Option<Vec<McpToolNode>>>>,
    /// Which groups are expanded (by name).
    pub expanded_groups: std::collections::HashSet<String>,
    /// Scroll state for the content area (shared, like messages_panel).
    pub scroll_state: Arc<RwLock<ScrollViewState>>,
    pub group_rects: std::cell::RefCell<Vec<(String, Rect)>>,
}

/// A tool node in the tree — either a group or a leaf tool.
#[derive(Clone, Debug)]
pub struct McpToolNode {
    pub name: String,
    pub description: String,
    pub children: Vec<McpToolNode>,
    pub is_leaf: bool,
}

impl Default for McpPanel {
    fn default() -> Self {
        Self { cards: Vec::new(), add_button_rect: None, add_button_pressed: false, card_rects: Vec::new(), tool_tree: std::sync::Arc::new(std::sync::RwLock::new(None)), expanded_groups: std::collections::HashSet::new(), scroll_state: Arc::new(RwLock::new(ScrollViewState::new())), group_rects: std::cell::RefCell::new(Vec::new()) }
    }
}

impl McpPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_card(&mut self, name: String, transport: String, endpoint: String, connected: bool, tools_count: usize) {
        // Don't add duplicates
        if self.cards.iter().any(|c| c.name == name && c.endpoint == endpoint) {
            return;
        }
        self.cards.push(McpCard {
            name,
            transport,
            endpoint,
            connected,
            tools_count,
            expanded: false,
        });
    }

    /// Returns Some(card_index) if a card was clicked, or None.
    /// If the "+" button was clicked, returns None (check handle_mouse for that).
    pub fn handle_card_click(&self, me: &crossterm::event::MouseEvent) -> Option<usize> {
        use crossterm::event::MouseEventKind;
        if let MouseEventKind::Down(_) = me.kind {
            for (i, rect) in self.card_rects.iter().enumerate() {
                if me.column >= rect.x && me.column < rect.x + rect.width
                    && me.row >= rect.y && me.row < rect.y + rect.height
                {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Returns the endpoint of a card for querying.
    pub fn card_endpoint(&self, index: usize) -> Option<&str> {
        self.cards.get(index).map(|c| c.transport.as_str())
    }

    /// Toggle expanded state of a card.
    pub fn toggle_card(&mut self, index: usize) {
        if let Some(card) = self.cards.get_mut(index) {
            card.expanded = !card.expanded;
        }
    }

    /// Toggle a group's expanded state by name.
    pub fn toggle_group(&mut self, group_name: &str) {
        if self.expanded_groups.contains(group_name) {
            self.expanded_groups.remove(group_name);
        } else {
            self.expanded_groups.insert(group_name.to_string());
        }
    }

    /// Handle group row click. Returns true if a group was clicked.
    pub fn handle_group_click(&self, me: &crossterm::event::MouseEvent) -> Option<String> {
        use crossterm::event::MouseEventKind;
        if let MouseEventKind::Down(_) = me.kind {
            for (name, rect) in self.group_rects.borrow().iter() {
                if me.column >= rect.x && me.column < rect.x + rect.width
                    && me.row >= rect.y && me.row < rect.y + rect.height
                {
                    return Some(name.clone());
                }
            }
        }
        None
    }

    /// Scroll up by one line.
    pub fn scroll_up(&self) {
        let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
        state.scroll_up();
    }

    /// Scroll down by one line.
    pub fn scroll_down(&self) {
        let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
        state.scroll_down();
    }

    /// Returns true if the "+" button was clicked. Also tracks pressed state for color swap.
    pub fn handle_mouse(&mut self, me: &crossterm::event::MouseEvent) -> bool {
        use crossterm::event::MouseEventKind;
        let on_button = |me: &crossterm::event::MouseEvent| -> bool {
            if let Some(rect) = self.add_button_rect {
                return me.column >= rect.x && me.column < rect.x + rect.width
                    && me.row >= rect.y && me.row < rect.y + rect.height;
            }
            false
        };
        match me.kind {
            MouseEventKind::Down(_) if on_button(me) => {
                self.add_button_pressed = true;
                true
            }
            MouseEventKind::Up(_) => {
                let was_pressed = self.add_button_pressed;
                self.add_button_pressed = false;
                was_pressed && on_button(me)
            }
            _ => false,
        }
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        if area.width < 3 || area.height < 4 {
            return;
        }

        let left = area.x;
        let right = area.x + area.width - 1;
        let top = area.y;
        let bottom = area.y + area.height - 1;

        let header_y = top;
        let header_text = " MCP ";
        let header_len = header_text.len() as u16;
        let header_x = area.x + (area.width.saturating_sub(header_len)) / 2;

        // ── Draw borders, header, and "+" button into the frame buffer ─
        {
            let buf = frame.buffer_mut();
            let mut put = |x: u16, y: u16, ch: char, fg: Color, bg: Color| {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.set_char(ch).set_fg(fg).set_bg(bg);
                }
            };

            // Fill background (skip top row — header overwrites it)
            for y in top + 1..=bottom {
                for x in left..=right {
                    put(x, y, ' ', PANEL_BG, PANEL_BG);
                }
            }
            // LEFT edge
            for y in top..bottom {
                put(left, y, LEFT, BORDER_FG, PANEL_BG);
            }
            // BOTTOM edge
            for x in left + 1..right {
                put(x, bottom, BOTTOM, BORDER_FG, PANEL_BG);
            }
            // RIGHT edge
            for y in top..bottom {
                put(right, y, RIGHT, PANEL_BG, BORDER_FG);
            }
            // Bottom corners
            put(left, bottom, CORNER_BL, PANEL_BG, BORDER_FG);
            put(right, bottom, CORNER_BR, PANEL_BG, BORDER_FG);

            // Header row IS the top row — extends to outer edges, closing the panel
            // Fill entire top row with header background, reaching the outer edges
            for x in left..=right {
                put(x, header_y, ' ', HEADER_BG, HEADER_BG);
            }
            // Header corner accents: upper-left quadrant on left, upper-right on right
            let corner_fg = Color::Rgb(0, 60, 60);      // dark teal
            let corner_bg = Color::Rgb(0, 90, 90);      // mildly dark teal (darker than header)
            put(left, header_y, '\u{2598}', corner_fg, corner_bg);   // ▘ upper-left
            put(right, header_y, '\u{259D}', corner_fg, corner_bg);  // ▝ upper-right

            // "+" button on bottom content row — outer cells are quartile corners
            let btn_text = " + ";
            let btn_len = btn_text.len() as u16;
            let btn_x = left + 1 + ((right - left - 1).saturating_sub(btn_len)) / 2;
            let btn_y = bottom - 1;
            self.add_button_rect = Some(Rect { x: btn_x, y: btn_y, width: btn_len, height: 1 });
            // Middle cell stays original color, outer cells get quartile treatment
            let btn_bg = Color::Rgb(100,20,100);   // BTN_BG from submit_button
            let q_light = Color::Rgb(150,20,150);  // BTN_LIGHT — 1/4 quartile fg
            let (q_fg, q_bg) = if self.add_button_pressed {
                (btn_bg, q_light) // swapped on click
            } else {
                (q_light, btn_bg) // normal: light quartile on dark
            };
            // Left corner cell: ▘ upper-left quartile
            put(btn_x, btn_y, '\u{2598}', q_fg, q_bg);
            // Middle "+" cell — unchanged from original
            put(btn_x + 1, btn_y, '+', HEADER_FG, btn_bg);
            // Right corner cell: ▝ upper-right quartile
            put(btn_x + 2, btn_y, '\u{259D}', q_fg, q_bg);
        } // buf dropped here

        // Write header text with bold via set_string (put closure can't handle modifiers)
        frame.buffer_mut().set_string(header_x, header_y, header_text, Style::default().fg(HEADER_FG).bg(HEADER_BG).add_modifier(Modifier::BOLD));

        // ── Content area: below header + 1 cell margin, above button row ──
        let content_area = Rect {
            x: left + 1,
            y: top + 2,
            width: area.width.saturating_sub(2),
            height: area.height.saturating_sub(4),
        };

        if self.cards.is_empty() {
            let text = Paragraph::new("No MCP servers connected.")
                .style(Style::default().fg(DIM_FG).bg(PANEL_BG));
            frame.render_widget(text, content_area);
            return;
        }

        // Alternating row backgrounds: first card opposite of header (magenta), then header color (teal)
        let row_colors = [Color::Rgb(100, 20, 100), HEADER_BG];
        let content_w = content_area.width;

        // Calculate total canvas height
        let mut canvas_h: u16 = 0;
        for card in &self.cards {
            canvas_h += 1; // card header
            if card.expanded {
                let tree = self.tool_tree.read().unwrap_or_else(|p| p.into_inner());
                match *tree {
                    Some(ref nodes) if !nodes.is_empty() => {
                        for node in nodes {
                            canvas_h += 1; // group row
                            if self.expanded_groups.contains(&node.name) {
                                canvas_h += node.children.len() as u16;
                            }
                        }
                    }
                    _ => canvas_h += 1, // "Loading..." or "No tools"
                }
            }
            canvas_h += 1; // gap between cards
        }
        canvas_h = canvas_h.max(content_area.height);

        // Read scroll state
        let state = self.scroll_state.read().unwrap_or_else(|p| p.into_inner());
        let current_offset = state.offset();
        drop(state);

        let scroll_size = Size::new(content_w, canvas_h);
        let mut scroll_view = ScrollView::new(scroll_size)
            .vertical_scrollbar_visibility(ScrollbarVisibility::Never)
            .horizontal_scrollbar_visibility(ScrollbarVisibility::Never);

        // Fill canvas background
        {
            let canvas_buf = scroll_view.buf_mut();
            for y in 0..canvas_h {
                for x in 0..content_w {
                    if let Some(cell) = canvas_buf.cell_mut((x, y)) {
                        cell.set_char(' ').set_fg(PANEL_BG).set_bg(PANEL_BG);
                    }
                }
            }
        }

        // Render cards into canvas at canvas-relative coordinates
        self.card_rects.clear();
        self.group_rects.borrow_mut().clear();
        let mut canvas_y: u16 = 0;
        let mut row_idx: usize = 0;

        for card in &self.cards {
            let card_h: u16 = if card.expanded {
                let tree = self.tool_tree.read().unwrap_or_else(|p| p.into_inner());
                match *tree {
                    Some(ref nodes) if !nodes.is_empty() => {
                        let mut h = 1u16;
                        for node in nodes {
                            h += 1;
                            if self.expanded_groups.contains(&node.name) {
                                h += node.children.len() as u16;
                            }
                        }
                        h
                    }
                    _ => 2,
                }
            } else { 1u16 };

            // Convert canvas y to screen y for click detection
            let screen_y = content_area.y as i32 + canvas_y as i32 - current_offset.y as i32;
            if screen_y + card_h as i32 > content_area.y as i32 && screen_y < (content_area.y + content_area.height) as i32 {
                let clip_top = (content_area.y as i32 - screen_y).max(0) as u16;
                let visible_h = card_h.saturating_sub(clip_top).min(content_area.height);
                let screen_rect = Rect {
                    x: content_area.x,
                    y: (screen_y + clip_top as i32).max(0) as u16,
                    width: content_w,
                    height: visible_h,
                };
                self.card_rects.push(screen_rect);
            }

            // Render card into canvas
            let card_area = Rect { x: 0, y: canvas_y, width: content_w, height: card_h };
            self.render_card_into(&mut scroll_view, card, card_area, row_colors[row_idx % 2]);
            row_idx += 1;
            canvas_y += card_h + 1;
        }

        // Convert group_rects from canvas coords to screen coords for click detection
        let offset_y = current_offset.y;
        let content_y = content_area.y;
        let content_x = content_area.x;
        let content_bottom = content_area.y + content_area.height;
        let mut group_rects = self.group_rects.borrow_mut();
        group_rects.retain(|(_, rect)| {
            let screen_y = content_y as i32 + rect.y as i32 - offset_y as i32;
            screen_y >= content_y as i32 && screen_y < content_bottom as i32
        });
        for (_, rect) in group_rects.iter_mut() {
            rect.y = (content_y as i32 + rect.y as i32 - offset_y as i32).max(0) as u16;
            rect.x = content_x + rect.x;
        }

        // Render the ScrollView into content_area
        let mut scroll_state = ScrollViewState::with_offset((0, current_offset.y).into());
        frame.render_stateful_widget(&scroll_view, content_area, &mut scroll_state);
        // Persist updated scroll state
        let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
        *state = scroll_state;
    }

    fn render_card_into(&self, scroll_view: &mut ScrollView, card: &McpCard, area: Rect, row_bg: Color) {
        scroll_view.render_widget(
            Block::default().style(Style::default().bg(row_bg)),
            area,
        );

        let status_icon = if card.connected { "\u{25CF}" } else { "\u{25CB}" };
        let status_color = if card.connected { CONNECTED_FG } else { DISCONNECTED_FG };
        let prefix = if card.expanded { "\u{25BE} " } else { "\u{25B8} " };

        let max_name = area.width.saturating_sub(prefix.chars().count() as u16 + 2);
        let display_name: String = if card.name.chars().count() > max_name as usize {
            format!("{}…", card.name.chars().take((max_name as usize).saturating_sub(1)).collect::<String>())
        } else {
            card.name.clone()
        };

        let title_text = format!("{}{}", prefix, display_name);
        let title_style = Style::default().fg(TEXT_FG).bg(row_bg).add_modifier(Modifier::BOLD);
        let buf = scroll_view.buf_mut();
        buf.set_string(area.x, area.y, &title_text, title_style);

        let status_x = area.x + area.width - 2;
        let status_style = Style::default().fg(status_color).bg(row_bg);
        buf.set_string(status_x, area.y, status_icon, status_style);

        if !card.expanded {
            return;
        }

        let inner_x = area.x + 1;
        let inner_w = area.width.saturating_sub(1);
        let mut dy = area.y + 1;

        let tree = self.tool_tree.read().unwrap_or_else(|p| p.into_inner());
        match *tree {
            Some(ref nodes) if !nodes.is_empty() => {
                for node in nodes {
                    if dy >= area.y + area.height { break; }

                    let is_expanded = self.expanded_groups.contains(&node.name);
                    let group_icon = if is_expanded { "\u{25BE}" } else { "\u{25B8}" };
                    let group_label = format!("{} {}", group_icon, node.name);

                    self.group_rects.borrow_mut().push((node.name.clone(), Rect { x: inner_x, y: dy, width: inner_w, height: 1 }));

                    buf.set_string(inner_x, dy, &group_label, Style::default().fg(TEXT_FG).bg(row_bg));
                    dy += 1;

                    if is_expanded {
                        for child in &node.children {
                            if dy >= area.y + area.height { break; }
                            let child_icon = if child.is_leaf { "\u{25B8}" } else { "\u{25BE}" };
                            let child_label = format!("  {} {}", child_icon, child.name);
                            buf.set_string(inner_x, dy, &child_label, Style::default().fg(DIM_FG).bg(row_bg));
                            dy += 1;
                        }
                    }
                }
            }
            Some(ref _nodes) => {
                buf.set_string(inner_x, dy, "  No tools available", Style::default().fg(DIM_FG).bg(row_bg));
            }
            None => {
                buf.set_string(inner_x, dy, "  Loading tools...", Style::default().fg(DIM_FG).bg(row_bg));
            }
        }
    }
}


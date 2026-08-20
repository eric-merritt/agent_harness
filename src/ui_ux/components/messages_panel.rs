// Messages panel — scrollable conversation history.
//
// Uses tui-scrollview so messages render at natural height and scroll.
// Each message: [Role Time ] [Content --wraps--]

use std::sync::{Arc, RwLock};

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Margin, Position, Rect, Size};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use tui_scrollview::{ScrollView, ScrollViewState, ScrollbarVisibility};

use crate::messaging::message::{Message, Role};

/// Near-black used as component background so panels pop off the gray terminal.
const PANEL_BG: Color = Color::Rgb(16, 16, 16);
/// Light text on the dark panel.
const TEXT_FG: Color = Color::Rgb(220, 220, 220);
/// Border color — neutral gray.
const BORDER_FG: Color = Color::Rgb(120, 120, 120);
/// User message label — dark teal.
const USER_FG: Color = Color::Rgb(0, 128, 128);
/// Agent message accent.
const AGENT_FG: Color = Color::Rgb(180, 140, 220);
/// Neon green for timestamps.
const TIME_FG: Color = Color::Rgb(57, 255, 20);

pub struct MessagesPanel {
    /// Shared reference to the message list.  Updated via sync().
    messages: Arc<RwLock<Vec<Message>>>,
    /// Shared scroll state so cloned instances persist scroll position across renders.
    scroll_state: Arc<RwLock<ScrollViewState>>,
    /// Track scrollbar drag state so click-drag works on the thumb.
    scrollbar_drag: Option<ScrollbarDrag>,
}

/// State for an in-progress scrollbar thumb drag.
struct ScrollbarDrag {
    /// Row (absolute terminal coordinate) where the drag started.
    /// Total number of scrollable rows (virtual height − viewport height).
    scrollable_rows: u16,
}

impl Default for MessagesPanel {
    fn default() -> Self {
        Self {
            messages: Arc::new(RwLock::new(Vec::new())),
            scroll_state: Arc::new(RwLock::new(ScrollViewState::new())),
            scrollbar_drag: None,
        }
    }
}

impl MessagesPanel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the internal message list with a snapshot from the conversation.
    pub fn sync(&mut self, snapshot: &[Message]) {
        let mut guard = self.messages.write().unwrap_or_else(|p| p.into_inner());
        *guard = snapshot.to_vec();
    }

    /// Handle a mouse event. Returns true if consumed.
    ///
    /// The scrollbar occupies the rightmost column of `area`.
    /// Clicking the track above/below the thumb scrolls one page.
    /// Clicking and dragging the thumb scrolls proportionally.
    pub fn handle_mouse(&mut self, me: &crossterm::event::MouseEvent, area: Rect) -> bool {
        use crossterm::event::MouseEventKind;

        // Only act on events inside this panel's area
        let in_area = me.row >= area.y
            && me.row < area.y + area.height
            && me.column >= area.x
            && me.column < area.x + area.width;

        if !in_area {
            // If we were dragging the scrollbar and the mouse left the area, release.
            if matches!(me.kind, MouseEventKind::Up(_)) {
                self.scrollbar_drag = None;
            }
            return false;
        }

        // The scrollbar lives in the rightmost column of the area.
        let scrollbar_col = area.x + area.width - 1;
        let on_scrollbar = me.column == scrollbar_col;

        match me.kind {
            MouseEventKind::ScrollUp => {
                let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
                state.scroll_up();
                true
            }
            MouseEventKind::ScrollDown => {
                let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
                state.scroll_down();
                true
            }
            MouseEventKind::Down(_) if on_scrollbar => {
                // Start a drag on the scrollbar thumb or page navigation on the track.
                self.begin_scrollbar_drag(me.row, area);
                true
            }
            MouseEventKind::Up(_) => {
                // Release any in-progress drag.
                self.scrollbar_drag = None;
                true
            }
            MouseEventKind::Moved if on_scrollbar => {
                // Continue dragging the thumb.
                self.continue_scrollbar_drag(me.row, area);
                true
            }
            _ => false,
        }
    }

    /// Begin a scrollbar interaction: thumb drag or track click.
    fn begin_scrollbar_drag(&mut self, mouse_row: u16, area: Rect) {
        // Compute the scrollable range.
        // We estimate virtual height from the message count — each message is at least 2 rows.
        let messages = self.messages.read().unwrap_or_else(|p| p.into_inner());
        let virtual_height: u16 = messages.len().max(1) as u16 * 2;
        let viewport_height = area.height.saturating_sub(1); // subtract horizontal scrollbar row
        let scrollable_rows = virtual_height.saturating_sub(viewport_height);

        if scrollable_rows == 0 {
            return;
        }

        // Row relative to the scrollbar track (0 = top of track).
        let track_top = area.y;
        let track_height = area.height.saturating_sub(1); // leave room for bottom scrollbar
        let rel_row = (mouse_row.saturating_sub(track_top)).min(track_height.saturating_sub(1));

        // Map the relative row to a scroll offset.
        let new_offset = (rel_row as f64 / track_height.max(1) as f64 * scrollable_rows as f64)
            .round() as u16;

        let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
        let offset = state.offset();
        state.set_offset(Position::new(offset.x, new_offset));
        drop(state);

        self.scrollbar_drag = Some(ScrollbarDrag {
            scrollable_rows,
        });
    }

    /// Continue dragging the scrollbar thumb.
    fn continue_scrollbar_drag(&mut self, mouse_row: u16, area: Rect) {
        let Some(drag) = self.scrollbar_drag.take() else {
            return;
        };

        // Recompute from scratch using the current mouse position relative to the track.
        let track_top = area.y;
        let track_height = area.height.saturating_sub(1); // leave room for bottom scrollbar
        let rel_row = (mouse_row.saturating_sub(track_top)).min(track_height.saturating_sub(1));

        let new_offset =
            (rel_row as f64 / track_height.max(1) as f64 * drag.scrollable_rows as f64)
                .round() as u16;

        let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
        let offset = state.offset();
        state.set_offset(Position::new(offset.x, new_offset));

        // Update start_row so repeated Moved events keep tracking.
        self.scrollbar_drag = Some(ScrollbarDrag {
            scrollable_rows: drag.scrollable_rows,
        });
    }

    /// Handle a key event for scrolling. Returns true if the key was consumed.
    pub fn handle_key(&self, key: &crossterm::event::KeyEvent) -> bool {
        use crossterm::event::KeyCode;
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
                state.scroll_up();
                true
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
                state.scroll_down();
                true
            }
            KeyCode::PageUp => {
                let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
                state.scroll_page_up();
                true
            }
            KeyCode::PageDown | KeyCode::Char(' ') => {
                let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
                state.scroll_page_down();
                true
            }
            KeyCode::Home => {
                let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
                state.scroll_to_top();
                true
            }
            KeyCode::End => {
                let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
                state.scroll_to_bottom();
                true
            }
            _ => false,
        }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        if area.width < 3 || area.height < 2 {
            return;
        }

        let buf = frame.buffer_mut();
        let left = area.x;
        let right = area.x + area.width - 1;
        let top = area.y;
        let bottom = area.y + area.height - 1;

        let mut put = |x: u16, y: u16, ch: char, fg: Color, bg: Color| {
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.set_char(ch).set_fg(fg).set_bg(bg);
            }
        };

        // Fill background
        for y in top..=bottom {
            for x in left..=right {
                put(x, y, ' ', PANEL_BG, PANEL_BG);
            }
        }

        // TOP edge
        for x in left + 1..right {
            put(x, top, '▔', BORDER_FG, PANEL_BG);
        }
        // LEFT edge
        for y in top + 1..bottom {
            put(left, y, '▎', BORDER_FG, PANEL_BG);
        }
        // BOTTOM edge
        for x in left + 1..right {
            put(x, bottom, '▁', BORDER_FG, PANEL_BG);
        }
        // RIGHT edge — ▊ draws using BG color, so swap fg/bg
        for y in top + 1..bottom {
            put(right, y, '▊', PANEL_BG, BORDER_FG);
        }
        // Corners
        put(left, top, '▗', PANEL_BG, BORDER_FG);
        put(right, top, '▖', PANEL_BG, BORDER_FG);
        put(left, bottom, '▝', PANEL_BG, BORDER_FG);
        put(right, bottom, '▘', PANEL_BG, BORDER_FG);

        // Content area inside the border (1-cell inset for the border chars).
        let content_area = area.inner(Margin { horizontal: 1, vertical: 1 });

        let messages = self.messages.read().unwrap_or_else(|p| p.into_inner());

        if messages.is_empty() {
            return;
        }

        // The scrollbar steals 1 column from the right edge of the content area,
        // plus 2 extra cells of right padding so text doesn't butt against the border.
        let scrollbar_width = 1;
        let scrollview_width = content_area.width.saturating_sub(scrollbar_width + 2);

        // Calculate line counts for each message: 1 header line + wrapped content lines.
        // The content column width is scrollview_width minus the prefix width.
        // "[ Agent ] 09:00pm " = 18 chars + buffer
        let prefix_width = 20;

        let msg_layouts: Vec<(u16, String)> = messages
            .iter()
            .map(|msg| {
                let role_label = msg.role.to_string();
                let time_label = msg.local_time();
                let prefix = format!("{} {}", role_label, time_label);
                let this_prefix_width = (prefix.len() as u16).max(prefix_width);
                let actual_content_width = scrollview_width.saturating_sub(this_prefix_width);

                // Count how many lines the content wraps to.
                let lines = if msg.content.is_empty() {
                    1
                } else {
                    let content_len = msg.content.chars().count();
                    let cw = actual_content_width.max(1) as usize;
                    1 + (content_len - 1) / cw // header line + wrapped lines
                };
                (lines as u16, prefix)
            })
            .collect();

        // Sum message heights + 1-row gap between each pair of messages
        let virtual_height: u16 = msg_layouts.iter().map(|(h, _)| *h).sum::<u16>()
            + messages.len().saturating_sub(1) as u16;

        // Canvas height = estimated virtual height, capped at a reasonable max to avoid huge gaps
        // from overcounting wrapped lines on unbreakable strings (e.g. file paths).
        let viewport_h = content_area.height;
        let canvas_h = virtual_height.min(viewport_h * 4).max(viewport_h + 1);
        let scroll_size = Size::new(scrollview_width, canvas_h);
        let mut scroll_view = ScrollView::new(scroll_size)
            .vertical_scrollbar_visibility(ScrollbarVisibility::Never)
            .horizontal_scrollbar_visibility(ScrollbarVisibility::Never);

        // Fill the scrollview's internal canvas buffer with PANEL_BG so that
        // cells widgets don't write to won't show terminal background.
        {
            let canvas_buf = scroll_view.buf_mut();
            let area = canvas_buf.area;
            for y in area.y..area.y + area.height {
                for x in area.x..area.x + area.width {
                    if let Some(cell) = canvas_buf.cell_mut((x, y)) {
                        cell.set_char(' ').set_fg(PANEL_BG).set_bg(PANEL_BG);
                    }
                }
            }
        }

        // Render messages into the canvas using canvas-relative coordinates (0-based).
        let mut y_offset: u16 = 0;
        for (i, msg) in messages.iter().enumerate() {
            let role_label = msg.role.to_string();
            let time_label = msg.local_time();
            let prefix = format!("{} {}", role_label, time_label);
            let this_prefix_width = (prefix.len() as u16).max(prefix_width);

            let msg_area = Rect {
                x: 0,
                y: y_offset,
                width: scrollview_width,
                height: msg_layouts[i].0,
            };

            let columns = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(this_prefix_width),
                    Constraint::Min(1),
                ])
                .split(msg_area);

            let role_color = match msg.role {
                Role::User => USER_FG,
                Role::Agent => AGENT_FG,
            };
            let prefix_line = Line::from(vec![
                Span::styled(role_label, Style::default().fg(role_color).bg(PANEL_BG)),
                Span::styled(time_label, Style::default().fg(TIME_FG).bg(PANEL_BG)),
            ]);
            let prefix_para = Paragraph::new(prefix_line);
            scroll_view.render_widget(prefix_para, columns[0]);

            let content_para = Paragraph::new(&*msg.content)
                .style(Style::default().fg(TEXT_FG).bg(PANEL_BG))
                .wrap(Wrap { trim: false });
            scroll_view.render_widget(content_para, columns[1]);

            y_offset += msg_layouts[i].0 + 1; // +1 for gap between messages
        }

        // Render the canvas into the content area (no scrollbar — we draw it manually).
        let state = self.scroll_state.read().unwrap_or_else(|p| p.into_inner());
        let current_offset = state.offset();
        drop(state);
        let mut scroll_state = ScrollViewState::with_offset((0, current_offset.y).into());
        frame.render_stateful_widget(&scroll_view, content_area, &mut scroll_state);
        // Persist the updated scroll state back
        let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
        *state = scroll_state;

        // Draw the vertical scrollbar manually in the rightmost column of content_area.
        // Fixed track height = area height - 4 (▲ top arrow + 1 margin + track + 1 margin + ▼ bottom arrow).
        let scrollbar_col = content_area.x + content_area.width - 1;
        let arrow_top = content_area.y;
        let arrow_bottom = content_area.y + content_area.height - 1;
        let track_top = content_area.y + 1;
        let track_bottom = content_area.y + content_area.height - 2;
        let track_height = track_bottom.saturating_sub(track_top);
        let scrollable = canvas_h.saturating_sub(content_area.height);

        if track_height > 0 && scrollable > 0 {
            let max_offset = scrollable as f64;
            let thumb_height = ((track_height as f64 * content_area.height as f64) / canvas_h as f64).max(1.0) as u16;
            let thumb_top = (current_offset.y as f64 / max_offset * (track_height - thumb_height) as f64).round() as u16;
            let thumb_top_abs = track_top + thumb_top;
            let thumb_bottom_abs = (thumb_top_abs + thumb_height).min(track_bottom);

            let mut put = |x: u16, y: u16, ch: char, fg: Color, bg: Color| {
                if let Some(cell) = frame.buffer_mut().cell_mut((x, y)) {
                    cell.set_char(ch).set_fg(fg).set_bg(bg);
                }
            };

            // Arrow heads
            put(scrollbar_col, arrow_top, '▲', BORDER_FG, PANEL_BG);
            put(scrollbar_col, arrow_bottom, '▼', BORDER_FG, PANEL_BG);

            // Track
            for y in track_top..track_bottom {
                if y < thumb_top_abs || y >= thumb_bottom_abs {
                    put(scrollbar_col, y, '│', BORDER_FG, PANEL_BG);
                }
            }
            // Thumb
            let thumb_fg = Color::Rgb(160, 160, 160);
            for y in thumb_top_abs..thumb_bottom_abs {
                put(scrollbar_col, y, '█', thumb_fg, thumb_fg);
            }
        }
    }
}

impl Clone for MessagesPanel {
    fn clone(&self) -> Self {
        Self {
            messages: Arc::clone(&self.messages),
            scroll_state: Arc::clone(&self.scroll_state),
            scrollbar_drag: None, // drag state doesn't clone — it's per-interaction
        }
    }
}

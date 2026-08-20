// Tools panel — card-style blocks with collapsible detail dropdowns.
// Teal is used ONLY as a chunky background on the click-target row, never on text.

use std::sync::{Arc, RwLock};

use ratatui::Frame;
use ratatui::layout::{Rect, Size};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};
use tui_scrollview::{ScrollView, ScrollViewState, ScrollbarVisibility};

/// Near-black used as component background so panels pop off the gray terminal.
const PANEL_BG: Color = Color::Rgb(16, 16, 16);
/// Card background — slightly lighter so cards stand out from the panel.
/// Teal background for hovered/selected card row (the click target).
const HIGHLIGHT_BG: Color = Color::Rgb(0, 60, 60);
/// Light text on the dark panel — never teal.
const TEXT_FG: Color = Color::Rgb(220, 220, 220);
/// Border color — neutral gray.
const BORDER_FG: Color = Color::Rgb(120, 120, 120);
/// Header background — dark magenta, alternating with MCP's teal.
const HEADER_BG: Color = Color::Rgb(100, 20, 100);
/// Header text — white.
const HEADER_FG: Color = Color::Rgb(255, 255, 255);
/// Dim text for secondary info.
const DIM_FG: Color = Color::Rgb(160, 160, 160);
/// Required param highlight.
const REQ_FG: Color = Color::Rgb(220, 180, 0);

/// Box-drawing characters for borders
const LEFT: char = '\u{258E}';  // ▎ left one quarter block
const BOTTOM: char = '\u{2581}'; // ▁ lower one eighth block
const RIGHT: char = '\u{258A}';  // ▊ left three quarters block
const CORNER_BL: char = '\u{259D}'; // ▝ quadrant upper right
const CORNER_BR: char = '\u{2598}'; // ▘ quadrant upper left

/// Represents a tool card with optional expanded details.
#[derive(Clone, Debug, Default)]
pub struct ToolCard {
    pub name: String,
    pub description: String,
    pub required_params: Vec<String>,
    pub optional_params: Vec<String>,
    pub expanded: bool,
    /// True when the cursor is over this row — gets the teal bg highlight.
    pub hovered: bool,
}

pub struct ToolsPanel {
    pub cards: Vec<ToolCard>,
    pub focused_index: usize,
    pub scroll_state: Arc<RwLock<ScrollViewState>>,
}

impl Default for ToolsPanel {
    fn default() -> Self {
        Self { cards: Vec::new(), focused_index: 0, scroll_state: Arc::new(RwLock::new(ScrollViewState::new())) }
    }
}

impl ToolsPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn scroll_up(&self) {
        let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
        state.scroll_up();
    }

    pub fn scroll_down(&self) {
        let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
        state.scroll_down();
    }

    pub fn add_card(&mut self, name: String, description: String, required: Vec<String>, optional: Vec<String>) {
        self.cards.push(ToolCard {
            name,
            description,
            required_params: required,
            optional_params: optional,
            expanded: false,
            hovered: false,
        });
    }

    /// Move the cursor up/down. Returns true if the position changed.
    pub fn navigate(&mut self, direction: isize) -> bool {
        if self.cards.is_empty() {
            return false;
        }
        let len = self.cards.len() as isize;
        let new_idx = ((self.focused_index as isize) + direction + len) % len;
        if new_idx != self.focused_index as isize {
            self.cards[self.focused_index].hovered = false;
            self.focused_index = new_idx as usize;
            self.cards[self.focused_index].hovered = true;
            return true;
        }
        false
    }

    /// Toggle the focused card expanded/collapsed.
    pub fn toggle_focused(&mut self) {
        if !self.cards.is_empty() {
            self.cards[self.focused_index].expanded = !self.cards[self.focused_index].expanded;
        }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        if area.width < 3 || area.height < 4 {
            return;
        }

        let left = area.x;
        let right = area.x + area.width - 1;
        let top = area.y;
        let bottom = area.y + area.height - 1;

        let header_y = top;
        let header_text = " Tools ";
        let header_len = header_text.len() as u16;
        let header_x = area.x + (area.width.saturating_sub(header_len)) / 2;

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

            // Fill entire top row with header background, reaching the outer edges
            for x in left..=right {
                put(x, header_y, ' ', HEADER_BG, HEADER_BG);
            }
            // Header corner accents: upper-left quadrant on left, upper-right on right
            let corner_fg = Color::Rgb(50, 20, 50);   // BTN_BG — darkest magenta
            let corner_bg = Color::Rgb(80, 30, 80);    // BTN_DARK — mildly dark magenta
            put(left, header_y, '\u{2598}', corner_fg, corner_bg);   // ▘ upper-left
            put(right, header_y, '\u{259D}', corner_fg, corner_bg);  // ▝ upper-right
        }

        // Write header text with bold via set_string
        frame.buffer_mut().set_string(header_x, header_y, header_text, Style::default().fg(HEADER_FG).bg(HEADER_BG).add_modifier(Modifier::BOLD));


        // Content area: below header row + 1 cell margin, inside left/right borders, above bottom border
        let content_area = Rect {
            x: left + 1,
            y: header_y + 2,
            width: area.width.saturating_sub(2),
            height: area.height.saturating_sub(3),
        };

        if self.cards.is_empty() {
            let text = Paragraph::new("No tools registered yet.")
                .style(Style::default().fg(DIM_FG).bg(PANEL_BG));
            frame.render_widget(text, content_area);
            return;
        }

        let row_colors = [Color::Rgb(0, 128, 128), HEADER_BG];
        let content_w = content_area.width;

        // Calculate canvas height
        let mut canvas_h: u16 = 0;
        for card in &self.cards {
            canvas_h += if card.expanded {
                let params_count = (card.required_params.len() + card.optional_params.len()).min(10);
                4u16 + params_count as u16
            } else { 1u16 };
            canvas_h += 1; // gap
        }
        canvas_h = canvas_h.max(content_area.height);

        let state = self.scroll_state.read().unwrap_or_else(|p| p.into_inner());
        let current_offset = state.offset();
        drop(state);

        let scroll_size = Size::new(content_w, canvas_h);
        let mut scroll_view = ScrollView::new(scroll_size)
            .vertical_scrollbar_visibility(ScrollbarVisibility::Never)
            .horizontal_scrollbar_visibility(ScrollbarVisibility::Never);

        // Fill canvas background
        {
            let buf = scroll_view.buf_mut();
            for y in 0..canvas_h {
                for x in 0..content_w {
                    if let Some(cell) = buf.cell_mut((x, y)) {
                        cell.set_char(' ').set_fg(PANEL_BG).set_bg(PANEL_BG);
                    }
                }
            }
        }

        let mut canvas_y: u16 = 0;
        let mut row_idx: usize = 0;
        for card in self.cards.iter() {
            let card_h: u16 = if card.expanded {
                let params_count = (card.required_params.len() + card.optional_params.len()).min(10);
                4u16 + params_count as u16
            } else { 1u16 };

            let card_area = Rect { x: 0, y: canvas_y, width: content_w, height: card_h };
            self.render_card_into(&mut scroll_view, card, card_area, row_colors[row_idx % 2]);
            row_idx += 1;
            canvas_y += card_h + 1;
        }

        let mut scroll_state = ScrollViewState::with_offset((0, current_offset.y).into());
        frame.render_stateful_widget(&scroll_view, content_area, &mut scroll_state);
        let mut state = self.scroll_state.write().unwrap_or_else(|p| p.into_inner());
        *state = scroll_state;
    }


    fn render_card_into(&self, scroll_view: &mut ScrollView, card: &ToolCard, area: Rect, row_bg: Color) {
        let card_bg = if card.hovered { HIGHLIGHT_BG } else { row_bg };

        scroll_view.render_widget(
            Block::default().style(Style::default().bg(card_bg)),
            area,
        );

        let title_line = Line::from(Span::styled(
            &card.name,
            Style::default().fg(TEXT_FG),
        ));
        let title_para = Paragraph::new(title_line);
        scroll_view.render_widget(title_para, Rect { x: area.x, y: area.y, width: area.width, height: 1 });

        if !card.expanded {
            return;
        }

        let mut dy = area.y + 1;
        let inner_x = area.x + 1;
        let inner_w = area.width.saturating_sub(1);

        if dy < area.y + area.height && !card.description.is_empty() {
            let desc = Paragraph::new(Line::from(format!("  {}", card.description)))
                .style(Style::default().fg(DIM_FG).bg(card_bg))
                .wrap(Wrap { trim: true });
            scroll_view.render_widget(desc, Rect { x: inner_x, y: dy, width: inner_w, height: 1 });
            dy += 1;
        }

        if dy < area.y + area.height {
            if !card.required_params.is_empty() || !card.optional_params.is_empty() {
                scroll_view.render_widget(
                    Paragraph::new(Line::from("  Params:")).style(Style::default().fg(TEXT_FG).bg(card_bg)),
                    Rect { x: inner_x, y: dy, width: inner_w, height: 1 },
                );
                dy += 1;

                for param in &card.required_params {
                    if dy >= area.y + area.height { break; }
                    scroll_view.render_widget(
                        Paragraph::new(Line::from(format!("    [req] {}", param))).style(Style::default().fg(REQ_FG).bg(card_bg)),
                        Rect { x: inner_x, y: dy, width: inner_w, height: 1 },
                    );
                    dy += 1;
                }
                for param in &card.optional_params {
                    if dy >= area.y + area.height { break; }
                    scroll_view.render_widget(
                        Paragraph::new(Line::from(format!("    [opt] {}", param))).style(Style::default().fg(DIM_FG).bg(card_bg)),
                        Rect { x: inner_x, y: dy, width: inner_w, height: 1 },
                    );
                    dy += 1;
                }
            }
        }

        if dy < area.y + area.height {
            scroll_view.render_widget(
                Paragraph::new(Line::from("  ... (TBD: invoke / disable / configure)")).style(Style::default().fg(Color::Rgb(100, 100, 120)).bg(card_bg)),
                Rect { x: inner_x, y: dy, width: inner_w, height: 1 },
            );
        }
    }
}

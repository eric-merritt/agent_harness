// Input bar — text entry field with key handling, wrapping, and scrollback.

use std::sync::{Arc, RwLock};

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Widget;

/// Near-black used as component background so panels pop off the gray terminal.
const PANEL_BG: Color = Color::Rgb(16, 16, 16);
/// Light text on the dark panel.
const TEXT_FG: Color = Color::Rgb(220, 220, 220);
/// Border color — neutral gray.
const BORDER_FG: Color = Color::Rgb(120, 120, 120);
/// Muted placeholder text.
const PLACEHOLDER_FG: Color = Color::Rgb(100, 100, 100);
/// Max visible lines in the input area; older lines go to scrollback.
const MAX_VISIBLE_LINES: usize = 3;

/// Shared mutable input buffer so the main loop can mutate it
/// and the render pass can read it.
#[derive(Clone, Debug)]
pub struct InputState {
	pub buffer: String,
	pub cursor_pos: usize,
	pub is_focused: bool,
	pub cursor_visible: bool,
	/// Character underneath the cursor (for blink restore).
	/// Render tick counter — toggles cursor_visible every N ticks.
	pub blink_tick: u64,
}

impl InputState {
	pub fn new() -> Self {
		Self {
			buffer: String::new(),
			cursor_pos: 0,
			is_focused: true,
			cursor_visible: true,
			blink_tick: 0,
		}
	}

	/// Advance the blink tick. Call once per render frame.
	pub fn tick_blink(&mut self) {
		self.blink_tick += 1;
		// Toggle visibility every 8 ticks (~250ms at 30fps)
		self.cursor_visible = self.blink_tick % 8 < 4;
	}

	/// Set focus and start blinking immediately.
	pub fn focus(&mut self) {
		self.cursor_visible = true;
		self.blink_tick = 0;
	}

	/// Process a key event and return true if the key was consumed.
	pub fn handle_key(&mut self, event: &KeyEvent) -> bool {
		use crossterm::event::KeyCode;
		if !self.is_focused {
			return false;
		}

		match event.code {
			KeyCode::Char(c) => {
				self.cursor_visible = true;
				self.blink_tick = 0;
				self.buffer.insert(self.cursor_pos, c);
				self.cursor_pos += 1;
				true
			}
			KeyCode::Backspace => {
				if self.cursor_pos > 0 {
					self.cursor_pos -= 1;
					self.buffer.remove(self.cursor_pos);
				}
				true
			}
			KeyCode::Delete => {
				if self.cursor_pos < self.buffer.len() {
					self.buffer.remove(self.cursor_pos);
				}
				true
			}
			KeyCode::Left => {
				self.cursor_visible = true;
				self.blink_tick = 0;
				if self.cursor_pos > 0 {
					self.cursor_pos -= 1;
				}
				true
			}
			KeyCode::Right => {
				self.cursor_visible = true;
				self.blink_tick = 0;
				if self.cursor_pos < self.buffer.len() {
					self.cursor_pos += 1;
				}
				true
			}
			KeyCode::Home => {
				self.cursor_visible = true;
				self.blink_tick = 0;
				self.cursor_pos = 0;
				true
			}
			KeyCode::End => {
				self.cursor_visible = true;
				self.blink_tick = 0;
				self.cursor_pos = self.buffer.len();
				true
			}
			KeyCode::Enter => true,
			KeyCode::Esc => {
				self.is_focused = false;
				true
			}
			_ => false,
		}
	}

	/// Clear the buffer after submit.
	pub fn clear(&mut self) {
		self.buffer.clear();
		self.cursor_pos = 0;
	}

	/// Take the buffer contents and clear it (for submit).
	pub fn take(&mut self) -> String {
		let text = self.buffer.clone();
		self.clear();
		text
	}
}

pub struct InputBar {
	pub state: Arc<RwLock<InputState>>,
}

impl InputBar {
	pub fn new() -> Self {
		Self {
			state: Arc::new(RwLock::new(InputState::new())),
		}
	}

	/// Legacy render via Frame — used when called from chat_interface.
	pub fn render(&self, frame: &mut Frame, area: Rect) {
		let snapshot = {
			let mut state = self.state.write().unwrap_or_else(|p| p.into_inner());
			state.tick_blink();
			InputSnapshot {
				buffer: state.buffer.clone(),
				cursor_pos: state.cursor_pos,
				is_focused: state.is_focused,
				cursor_visible: state.cursor_visible,
			}
		};
		let widget = InputBarWidget { snapshot };
		frame.render_widget(widget, area);
	}
}

/// Immutable snapshot of input state for rendering.
struct InputSnapshot {
	buffer: String,
	cursor_pos: usize,
	is_focused: bool,
	cursor_visible: bool,
}

/// Wrap a string into lines of at most `wrap_width` characters.
/// Returns a vector of line strings (each ≤ wrap_width).
fn wrap_text(text: &str, wrap_width: usize) -> Vec<String> {
	let mut lines = Vec::new();
	let chars: Vec<char> = text.chars().collect();
	let len = chars.len();

	if len == 0 {
		return lines;
	}

	let mut start = 0;
	while start < len {
		let max_end = (start + wrap_width).min(len);
		let mut end = max_end;

		// If we didn't hit the end, try to break at a word boundary
		// going backward from max_end.
		if max_end < len {
			while end > start && !chars[end - 1].is_whitespace() {
				end -= 1;
			}
			// If no whitespace found, break at max_end anyway
			if end == start {
				end = max_end;
			}
		}

		let line: String = chars[start..end].iter().collect();
		lines.push(line);
		start = end;
	}

	lines
}

/// Given wrapped lines and a cursor position in the original string,
/// return (line_index, char_offset_within_line) for the cursor.
fn cursor_position_in_wrapped(wrapped: &[String], cursor_pos: usize) -> (usize, usize) {
	let mut pos = 0;
	for (i, line) in wrapped.iter().enumerate() {
		let line_len = line.chars().count();
		if pos + line_len > cursor_pos {
			return (i, cursor_pos - pos);
		}
		pos += line_len;
	}
	// Cursor is at the very end
	(wrapped.len(), 0)
}

/// One-shot widget that renders the input bar from a snapshot.
struct InputBarWidget {
	snapshot: InputSnapshot,
}

impl Widget for InputBarWidget {
	fn render(self, area: Rect, buf: &mut Buffer) {
		if area.width < 3 || area.height < 2 {
			return;
		}

		let (left, right) = (area.left(), area.right() - 1);
		let (top, bottom) = (area.top(), area.bottom() - 1);

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

		// Corners — top-left and bottom-left swap fg/bg (▎ draws with fg)
		put(left, top, '▗', PANEL_BG, BORDER_FG);
		put(right, top, '▖', PANEL_BG, BORDER_FG);
		put(left, bottom, '▝', PANEL_BG, BORDER_FG);
		put(right, bottom, '▘', PANEL_BG, BORDER_FG);

		// Inner content — wrap text and show last MAX_VISIBLE_LINES.
		let has_text = !self.snapshot.buffer.is_empty();
		let cursor = self.snapshot.cursor_pos;

		let text_fg = if has_text { TEXT_FG } else { PLACEHOLDER_FG };

		if has_text {
			// Wrap at inner width (area.width - 2 for left/right border cells)
			let inner_width = (right - left + 1).max(2) as usize;
			let wrap_width = inner_width.saturating_sub(2);
			let all_wrapped = wrap_text(&self.snapshot.buffer, wrap_width.max(1));

			// Only show the last MAX_VISIBLE_LINES lines (scrollback for older).
			let visible: Vec<String> = if all_wrapped.len() > MAX_VISIBLE_LINES {
				all_wrapped[all_wrapped.len() - MAX_VISIBLE_LINES..]
					.iter()
					.cloned()
					.collect()
			} else {
				all_wrapped.clone()
			};

			// Determine cursor position within the visible lines.
			let (cursor_line, _cursor_col) = cursor_position_in_wrapped(&all_wrapped, cursor);

			// Map cursor_line to the visible set.
			let total_wrapped = all_wrapped.len();
			let visible_start = if total_wrapped > MAX_VISIBLE_LINES {
				total_wrapped - MAX_VISIBLE_LINES
			} else {
				0
			};
			// Clamp cursor line to last visible line so end-of-buffer cursor always shows
			let effective_cursor_line = cursor_line.min(
				visible_start
					.saturating_add(visible.len())
					.saturating_sub(1),
			);

			let inner_left = left + 1;
			let inner_right = right - 1;

			for (i, line) in visible.iter().enumerate() {
				let y = top + 1 + (i as u16);
				if y >= bottom {
					break;
				}

				let abs_line_idx = visible_start + i;
				let is_cursor_line = effective_cursor_line >= visible_start
					&& effective_cursor_line < visible_start + visible.len()
					&& effective_cursor_line - visible_start == i;
				let cursor_col_in_this_line = if is_cursor_line {
					// Recalculate column within this specific line
					let chars_before: usize = all_wrapped[visible_start..abs_line_idx]
						.iter()
						.map(|l| l.chars().count())
						.sum();
					cursor.saturating_sub(chars_before)
				} else {
					usize::MAX
				};

				let mut x = inner_left;
				let line_char_count = line.chars().count();
				for ch in line.chars() {
					if x >= inner_right {
						break;
					}
					let is_at_cursor =
						is_cursor_line && cursor_col_in_this_line == (x - inner_left) as usize;
					let (ch, fg) = if is_at_cursor && self.snapshot.cursor_visible {
						('█', TEXT_FG)
					} else {
						(ch, text_fg)
					};
					put(x, y, ch, fg, PANEL_BG);
					x += 1;
				}

				// Draw cursor if it's past the end of this line's text — always full-width block, blink on/off
				if is_cursor_line && cursor_col_in_this_line >= line_char_count {
					let cursor_x = inner_left + cursor_col_in_this_line as u16;
					if cursor_x >= inner_left && cursor_x < inner_right {
						if self.snapshot.cursor_visible {
							put(cursor_x, y, '\u{2588}', TEXT_FG, PANEL_BG);
						} else {
							put(cursor_x, y, ' ', TEXT_FG, PANEL_BG);
						}
					}
				}
			}
		} else {
			// Placeholder text — no cursor
			let placeholder = " Type here... ";
			let inner_left = left + 1;
			let inner_right = right - 1;
			let mut x = inner_left;
			for ch in placeholder.chars() {
				if x >= inner_right {
					break;
				}
				put(x, top + 1, ch, PLACEHOLDER_FG, PANEL_BG);
				x += 1;
			}
		}
	}
}

impl Clone for InputBar {
	fn clone(&self) -> Self {
		Self {
			state: Arc::clone(&self.state),
		}
	}
}

impl Default for InputBar {
	fn default() -> Self {
		Self::new()
	}
}

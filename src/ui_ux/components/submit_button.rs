use ratatui::{
	buffer::Buffer,
	layout::Rect,
	style::{Color, Modifier, Style},
	widgets::Widget,
};

const BTN_BG: Color = Color::Rgb(150, 50, 150);
const BTN_LABEL: Color = Color::Rgb(255, 255, 255);
const BTN_LIGHT: Color = Color::Rgb(200, 40, 200);
const BTN_DARK: Color = Color::Rgb(100, 20, 100);

pub struct SubmitButton<'a> {
	pub label: &'a str,
	pub is_pressed: bool,
	pub is_hovered: bool,
}

/// Return the color to use for a border element, swapping LIGHT↔DARK when pressed.
/// BTN_BG is never changed — it represents the background fill.
fn border_color(pressed: bool, light: Color, dark: Color) -> (Color, Color) {
	if pressed {
		(dark, light) // swap: LIGHT→DARK, DARK→LIGHT
	} else {
		(light, dark)
	}
}

impl<'a> Widget for SubmitButton<'a> {
	fn render(self, area: Rect, buf: &mut Buffer) {
		if area.width < 3 || area.height < 3 {
			return;
		}

		let (left, right) = (area.left(), area.right() - 1);
		let (top, bottom) = (area.top(), area.bottom() - 1);

		let mut put = |x: u16, y: u16, ch: char, fg: Color, bg: Color| {
			if let Some(cell) = buf.cell_mut((x, y)) {
				cell.set_char(ch).set_fg(fg).set_bg(bg);
			}
		};

		// When pressed, swap LIGHT↔DARK on all border elements
		let (top_fg, right_bg) = border_color(self.is_pressed, BTN_LIGHT, BTN_DARK);
		// top/left edges use BTN_LIGHT fg, bottom/right edges use BTN_DARK
		// After swap: top/left get BTN_DARK, bottom/right get BTN_LIGHT

		// Face: fills everything between the borders, right up to them
		for y in top + 1..bottom {
			for x in left + 1..right {
				put(x, y, ' ', Color::Reset, BTN_BG);
			}
		}

		// TOP edge: light line (swaps to dark when pressed)
		for x in left + 1..right {
			put(x, top, '▔', top_fg, BTN_BG);
		}
		// LEFT edge: light line (swaps to dark when pressed)
		for y in top + 1..bottom {
			put(left, y, '▎', top_fg, BTN_BG);
		}
		// BOTTOM edge: dark line (swaps to light when pressed)
		for x in left + 1..right {
			put(x, bottom, '▁', right_bg, BTN_BG);
		}
		// RIGHT edge: dark line — fg=BTN_BG, bg=BTN_DARK (swaps to BTN_LIGHT when pressed)
		for y in top + 1..bottom {
			put(right, y, '▊', BTN_BG, right_bg);
		}

		// Corners — bg color swaps LIGHT↔DARK when pressed; fg stays BTN_BG
		put(left, top, '▗', BTN_BG, top_fg); // top-left: light (dark when pressed)
		put(right, top, '▖', BTN_BG, top_fg); // top-right: light (dark when pressed)
		put(left, bottom, '▝', BTN_BG, right_bg); // bottom-left: dark (light when pressed)
		put(right, bottom, '▘', BTN_BG, right_bg); // bottom-right: dark (light when pressed)

		// Label
		let inner_width = area.width.saturating_sub(2);
		let inner_height = area.height.saturating_sub(2);
		let label_len = self.label.len() as u16;
		let text_x = area.left() + 1 + (inner_width.saturating_sub(label_len)) / 2;
		let text_y = area.top() + 1 + inner_height / 2;

		let mut label_style = Style::default()
			.fg(BTN_LABEL)
			.bg(BTN_BG)
			.add_modifier(Modifier::BOLD);
		if self.is_hovered {
			label_style = label_style.add_modifier(Modifier::UNDERLINED);
		}

		if text_x + label_len <= area.right() - 1 && text_y < area.bottom() - 1 {
			buf.set_string(text_x, text_y, self.label, label_style);
		}
	}
}

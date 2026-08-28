use ratatui::layout::{Constraint, Layout, Rect};

/// Hard minimums so panels stay usable on very small terminals
const INPUT_MIN_HEIGHT: u16 = 5;
const TOOL_MIN_HEIGHT: u16 = 3;
const SIDEBAR_MIN_WIDTH: u16 = 10;
const BUTTON_MIN_WIDTH: u16 = 10;

/// Hard maximums so panels never dominate a fullscreen terminal
const INPUT_MAX_HEIGHT: u16 = 5;
const SIDEBAR_MAX_WIDTH_PCT: f64 = 0.15;

/// Minimum terminal width required to show the sidebar at all.
const SIDEBAR_MIN_TERMINAL_WIDTH: u16 = 80;

/// Resolved geometry for the default 4-panel layout.
///
///  ┌─────────────────────────┬──────────────┐
///  │                         │   Tools      │
///  │    Messages  (85 %)     │   (50 %)     │
///  │                         ├──────────────┤
///  ├─────────────────────────┤   MCP        │
///  │  Input (15 %, max 8 %)  │   (50 %)     │
///  │                         │              │
///  └─────────────────────────┴──────────────┘
#[derive(Clone, Copy, Debug)]
pub struct DefaultLayout {
	pub chat_column: Rect,
	pub messages: Rect,
	pub input: Rect,
	pub user_input: Rect,
	pub submit_btn: Rect,
	pub btn_container_bg: Rect,
	pub sidebar: Rect,
	pub tools: Rect,
	pub mcp: Rect,
}

impl DefaultLayout {
	/// Returns the full set of named areas for a given terminal rect.
	pub fn resolve(area: Rect) -> Self {
		// ── Sidebar width ──
		let sidebar_w = if area.width < SIDEBAR_MIN_TERMINAL_WIDTH {
			0
		} else {
			let raw = (area.width as f64 * 0.20).round() as u16;
			let mx = (area.width as f64 * SIDEBAR_MAX_WIDTH_PCT).round() as u16;
			raw.min(mx).max(SIDEBAR_MIN_WIDTH)
		};
		let chat_w = area.width.saturating_sub(sidebar_w);

		// ── Horizontal split: chat | sidebar ──
		let [chat_area, sidebar_area] =
			Layout::horizontal([Constraint::Length(chat_w), Constraint::Length(sidebar_w)])
				.areas(area);

		// ── Vertical split inside chat: messages | input ──
		// Input is 5 lines + 1-line margin above and below (7 total)
		let margin = 1;
		let input_h = ((area.height as f64 * 0.15).round() as u16)
			.min(INPUT_MAX_HEIGHT)
			.max(INPUT_MIN_HEIGHT)
			.min(area.height);
		let input_total_h = input_h + 2 * margin;
		let msg_h = chat_area
			.height
			.saturating_sub(input_total_h)
			.saturating_sub(margin);

		// Top margin between messages panel and chat column top edge.
		let [_msg_margin_top, messages_outer, input_outer] = Layout::vertical([
			Constraint::Length(margin),
			Constraint::Length(msg_h),
			Constraint::Length(input_total_h),
		])
		.areas(chat_area);

		// Shrink messages_area to exclude margin on left/right sides.
		let messages_area = ratatui::layout::Rect {
			x: messages_outer.x + margin,
			y: messages_outer.y,
			width: messages_outer.width.saturating_sub(2 * margin),
			height: messages_outer.height,
		};

		// Shrink input_area to exclude margin on all sides
		let hmargin = 1;
		let input_area = ratatui::layout::Rect {
			x: input_outer.x + hmargin,
			y: input_outer.y + margin,
			width: input_outer.width.saturating_sub(2 * hmargin),
			height: input_h,
		};

		// ── Horizontal split inside input: text | (1-cell gap) | button ──
		let btn_w = ((chat_w as f64 * 0.10).round() as u16)
			.max((chat_w as f64 * 0.12).round() as u16)
			.max(BUTTON_MIN_WIDTH)
			.min(chat_w);
		let gap = 1;
		let text_w = input_area.width.saturating_sub(btn_w).saturating_sub(gap);

		let [user_input_area, submit_btn_full] =
			Layout::horizontal([Constraint::Length(text_w), Constraint::Length(btn_w)])
				.areas(input_area);

		// Shift button right by the gap, leaving 1 cell between text and button
		let submit_btn_full = ratatui::layout::Rect {
			x: submit_btn_full.x + gap,
			y: submit_btn_full.y,
			width: submit_btn_full.width,
			height: submit_btn_full.height,
		};

		// ── 5-line button, centered vertically within the container ──
		let btn_h = 5.min(submit_btn_full.height);
		let btn_top_offset = (submit_btn_full.height.saturating_sub(btn_h)) / 2;
		let submit_btn_area = ratatui::layout::Rect {
			x: submit_btn_full.x,
			y: submit_btn_full.y + btn_top_offset,
			width: submit_btn_full.width,
			height: btn_h,
		};

		// ── Sidebar vertical split: 1 gap above tools, tools, 1 gap between, mcp, 1 gap below ──
		let gap = 1;
		let sidebar_content_h = sidebar_area.height.saturating_sub(3 * gap); // top + between + bottom
		let tools_h = if sidebar_w == 0 {
			0
		} else {
			(sidebar_content_h.saturating_div(2)).max(TOOL_MIN_HEIGHT)
		};
		let mcp_h = if sidebar_w == 0 {
			0
		} else {
			sidebar_content_h.saturating_sub(tools_h)
		};

		let [
			_tools_gap_top,
			tools_area,
			_tools_mcp_gap,
			mcp_area,
			_mcp_gap_bottom,
		] = Layout::vertical([
			Constraint::Length(gap),
			Constraint::Length(tools_h),
			Constraint::Length(gap),
			Constraint::Length(mcp_h),
			Constraint::Length(gap),
		])
		.areas(sidebar_area);

		// Shrink tools and mcp areas by 1 cell on the right for a gap from the sidebar edge.
		let tools_area = Rect {
			x: tools_area.x,
			y: tools_area.y,
			width: tools_area.width.saturating_sub(gap),
			height: tools_area.height,
		};
		let mcp_area = Rect {
			x: mcp_area.x,
			y: mcp_area.y,
			width: mcp_area.width.saturating_sub(gap),
			height: mcp_area.height,
		};

		Self {
			chat_column: chat_area,
			messages: messages_area,
			input: input_area,
			user_input: user_input_area,
			submit_btn: submit_btn_area,
			btn_container_bg: submit_btn_full,
			sidebar: sidebar_area,
			tools: tools_area,
			mcp: mcp_area,
		}
	}
}

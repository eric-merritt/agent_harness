// MCP Server Configuration Modal

use std::sync::{Arc, RwLock};

use ratatui::Frame;
use ratatui::layout::{Margin, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

const MODAL_BG: Color = Color::Rgb(120, 120, 120);
const MODAL_BORDER: Color = Color::Rgb(120, 120, 120);
const HEADER_BG: Color = Color::Rgb(0, 128, 128);
const HEADER_FG: Color = Color::Rgb(255, 255, 255);
const TEXT_FG: Color = Color::Rgb(220, 220, 220);
const LABEL_FG: Color = Color::Rgb(0, 128, 128);
const INPUT_BG: Color = Color::Rgb(28, 28, 28);
const BTN_SAVE_BG: Color = Color::Rgb(0, 128, 128);
const BTN_CANCEL_BG: Color = Color::Rgb(100, 20, 100);
const BTN_FG: Color = Color::Rgb(255, 255, 255);
const STATUS_OK: Color = Color::Rgb(0, 200, 100);
const STATUS_ERR: Color = Color::Rgb(220, 80, 80);
const STATUS_IDLE: Color = Color::Rgb(160, 160, 160);
const CHECKBOX_ON: char = '\u{25C9}';
const CHECKBOX_OFF: char = '\u{25CB}';

pub fn detect_transport(endpoint: &str) -> &'static str {
	let lower = endpoint.to_lowercase();
	if lower.starts_with("http://")
		|| lower.starts_with("https://")
		|| lower.starts_with("ws://")
		|| lower.starts_with("wss://")
	{
		if lower.contains("sse") || lower.contains("/sse") {
			"SSE"
		} else if lower.contains("stream") || lower.contains("streamable") {
			"Streamable HTTP"
		} else {
			"HTTP"
		}
	} else if lower.starts_with("grpc://") || lower.starts_with("grpc-") {
		"gRPC"
	} else if !endpoint.is_empty() {
		"Stdio"
	} else {
		"—"
	}
}

#[derive(Clone, Debug, Default)]
pub enum ConfigState {
	#[default]
	Idle,
	Connecting,
	Connected,
	Failed(String),
}

pub struct McpModal {
	pub server_name: String,
	pub endpoint: String,
	pub focused_field: FocusedField,
	pub requires_auth: bool,
	pub username: String,
	pub password: String,
	pub config_state: Arc<RwLock<ConfigState>>,
	/// Callback set by App to close the modal.
	save_rect: std::cell::RefCell<Option<Rect>>,
	cancel_rect: std::cell::RefCell<Option<Rect>>,
	auth_rect: std::cell::RefCell<Option<Rect>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum FocusedField {
	#[default]
	Name,
	Endpoint,
	Username,
	Password,
}

impl McpModal {
	pub fn new() -> Self {
		Self {
			server_name: String::new(),
			endpoint: String::new(),
			focused_field: FocusedField::default(),
			requires_auth: false,
			username: String::new(),
			password: String::new(),
			config_state: Arc::new(RwLock::new(ConfigState::default())),
			save_rect: std::cell::RefCell::new(None),
			cancel_rect: std::cell::RefCell::new(None),
			auth_rect: std::cell::RefCell::new(None),
		}
	}

	/// Open the modal — reset state and show.
	pub fn open(&mut self) {
		self.reset();
	}

	/// Close the modal — calls the on_close callback if set.

	/// Set the close callback.

	/// Start connection test. Returns the endpoint info for the caller to spawn async work.
	pub fn connect(&self) -> Option<(String, String, bool, String, String)> {
		if !self.is_valid() {
			return None;
		}
		let state = self.config_state.read().unwrap_or_else(|p| p.into_inner());
		if matches!(*state, ConfigState::Connecting) {
			return None;
		}
		drop(state);
		log::info!(
			"MCP connection test initiated: server={}, endpoint={}",
			self.server_name,
			self.endpoint
		);
		Some((
			self.server_name.clone(),
			self.endpoint.clone(),
			self.requires_auth,
			self.username.clone(),
			self.password.clone(),
		))
	}

	pub fn is_connected(&self) -> bool {
		matches!(
			*self.config_state.read().unwrap_or_else(|p| p.into_inner()),
			ConfigState::Connected
		)
	}

	pub fn is_valid(&self) -> bool {
		!self.server_name.trim().is_empty() && !self.endpoint.trim().is_empty()
	}

	pub fn reset(&mut self) {
		self.server_name.clear();
		self.endpoint.clear();
		self.requires_auth = false;
		self.username.clear();
		self.password.clear();
		*self.config_state.write().unwrap_or_else(|p| p.into_inner()) = ConfigState::Idle;
	}

	pub fn render(&self, frame: &mut Frame) {
		let full = frame.area();
		let state = self.config_state.read().unwrap_or_else(|p| p.into_inner());
		let is_busy = matches!(*state, ConfigState::Connecting);
		let is_valid = self.is_valid();

		let w = (full.width / 2).max(48).min(full.width.saturating_sub(4));
		let base_h = if self.requires_auth { 19u16 } else { 15u16 };
		let h = base_h.min(full.height.saturating_sub(4));
		let x = full.x + (full.width.saturating_sub(w)) / 2;
		let y = full.y + (full.height.saturating_sub(h)) / 2;
		let modal_area = Rect {
			x,
			y,
			width: w,
			height: h,
		};

		let block = Block::default()
			.borders(Borders::ALL)
			.border_style(Style::default().fg(MODAL_BORDER))
			.style(Style::default().bg(MODAL_BG));
		frame.render_widget(block, modal_area);

		let inner = modal_area.inner(Margin {
			horizontal: 1,
			vertical: 1,
		});
		let field_w = inner.width.saturating_sub(12);

		let header_text = match *state {
			ConfigState::Idle => " Add MCP Server ",
			ConfigState::Connecting => " Connecting... ",
			ConfigState::Connected => " Connected! ",
			ConfigState::Failed(_) => " Connection Failed ",
		};
		let header_bg = match *state {
			ConfigState::Connected => STATUS_OK,
			ConfigState::Failed(_) => STATUS_ERR,
			_ => HEADER_BG,
		};
		frame.render_widget(
			Paragraph::new(Line::from(Span::styled(
				header_text,
				Style::default().fg(HEADER_FG).bg(header_bg),
			))),
			Rect {
				x: inner.x,
				y: inner.y,
				width: inner.width,
				height: 1,
			},
		);

		let mut cy = inner.y + 2;
		let label_w = 10u16;

		let render_field = |frame: &mut Frame,
		                    label: &str,
		                    value: &str,
		                    placeholder: &str,
		                    field: &FocusedField,
		                    focused: &FocusedField,
		                    y: &mut u16| {
			let lbl = Paragraph::new(Line::from(Span::styled(
				label,
				Style::default().fg(LABEL_FG),
			)));
			frame.render_widget(
				lbl,
				Rect {
					x: inner.x,
					y: *y,
					width: label_w,
					height: 1,
				},
			);
			let ia = Rect {
				x: inner.x + label_w + 1,
				y: *y,
				width: field_w,
				height: 1,
			};
			let text = if value.is_empty() && field != focused {
				placeholder
			} else {
				value
			};
			let display: String = if field == &FocusedField::Password {
				value.chars().map(|_| '\u{2022}').collect()
			} else {
				text.to_string()
			};
			frame.render_widget(
				Paragraph::new(display.as_str()).style(Style::default().fg(TEXT_FG).bg(INPUT_BG)),
				ia,
			);
			if field == focused {
				let cx = ia.x + value.chars().count().min(field_w as usize - 1) as u16;
				if cx < ia.x + field_w {
					frame.render_widget(
						Paragraph::new("█").style(Style::default().fg(TEXT_FG)),
						Rect {
							x: cx,
							y: *y,
							width: 1,
							height: 1,
						},
					);
				}
			}
			*y += 2;
		};

		render_field(
			frame,
			"Name:",
			&self.server_name,
			"Enter server name...",
			&FocusedField::Name,
			&self.focused_field,
			&mut cy,
		);
		render_field(
			frame,
			"Endpoint:",
			&self.endpoint,
			"URL, command, or path...",
			&FocusedField::Endpoint,
			&self.focused_field,
			&mut cy,
		);

		cy -= 1;
		let detected = detect_transport(&self.endpoint);
		let det_text = if self.endpoint.is_empty() {
			""
		} else {
			&format!("Type: {}", detected)
		};
		let det_color = if self.endpoint.is_empty() {
			STATUS_IDLE
		} else {
			LABEL_FG
		};
		frame.render_widget(
			Paragraph::new(Line::from(Span::styled(
				det_text,
				Style::default().fg(det_color),
			)))
			.style(Style::default().bg(MODAL_BG)),
			Rect {
				x: inner.x,
				y: cy,
				width: inner.width,
				height: 1,
			},
		);
		cy += 2;

		let auth_x = inner.x;
		let auth_y = cy;
		let cb = if self.requires_auth {
			CHECKBOX_ON
		} else {
			CHECKBOX_OFF
		};
		let auth_line = Line::from(vec![
			Span::styled(
				format!("{} ", cb),
				Style::default().fg(if self.requires_auth {
					STATUS_OK
				} else {
					LABEL_FG
				}),
			),
			Span::styled("Require Auth", Style::default().fg(LABEL_FG)),
		]);
		self.auth_rect.replace(Some(Rect {
			x: auth_x,
			y: auth_y,
			width: 16,
			height: 1,
		}));
		frame.render_widget(
			Paragraph::new(auth_line).style(Style::default().bg(MODAL_BG)),
			Rect {
				x: auth_x,
				y: auth_y,
				width: field_w,
				height: 1,
			},
		);
		cy += 2;

		if self.requires_auth {
			render_field(
				frame,
				"Username:",
				&self.username,
				"Enter username...",
				&FocusedField::Username,
				&self.focused_field,
				&mut cy,
			);
			render_field(
				frame,
				"Password:",
				&self.password,
				"Enter password...",
				&FocusedField::Password,
				&self.focused_field,
				&mut cy,
			);
		}

		let status_y = cy;
		let status_text = match *state {
			ConfigState::Connecting => "Testing connection...",
			ConfigState::Connected => "Connection successful",
			ConfigState::Failed(_) => "Connection failed",
			ConfigState::Idle if !is_valid => "Fill in Name and Endpoint to continue",
			ConfigState::Idle => "Ready",
		};
		let status_color = match *state {
			ConfigState::Connecting => STATUS_IDLE,
			ConfigState::Connected => STATUS_OK,
			ConfigState::Failed(_) => STATUS_ERR,
			_ => STATUS_IDLE,
		};
		frame.render_widget(
			Paragraph::new(Line::from(Span::styled(
				status_text,
				Style::default().fg(status_color),
			)))
			.style(Style::default().bg(MODAL_BG)),
			Rect {
				x: inner.x,
				y: status_y,
				width: inner.width,
				height: 1,
			},
		);

		let btn_y = inner.y + inner.height - 2;
		let btn_w = 8u16;
		let gap = 2u16;
		let total_btn_w = btn_w * 2 + gap;
		let btn_start_x = inner.x + (inner.width.saturating_sub(total_btn_w)) / 2;

		let render_btn = |frame: &mut Frame, label: &str, bg: Color, area: Rect, disabled: bool| {
			let bg = if disabled { Color::Rgb(60, 60, 60) } else { bg };
			let fg = if disabled {
				Color::Rgb(100, 100, 100)
			} else {
				BTN_FG
			};
			let pad = (btn_w as usize).saturating_sub(label.len()) / 2;
			let mut chars: Vec<Span> = Vec::with_capacity(btn_w as usize);
			for _ in 0..pad {
				chars.push(Span::styled(" ", Style::default().fg(fg).bg(bg)));
			}
			for c in label.chars() {
				chars.push(Span::styled(c.to_string(), Style::default().fg(fg).bg(bg)));
			}
			while chars.len() < btn_w as usize {
				chars.push(Span::styled(" ", Style::default().fg(fg).bg(bg)));
			}
			frame.render_widget(Paragraph::new(Line::from(chars)), area);
		};

		let save_area = Rect {
			x: btn_start_x,
			y: btn_y,
			width: btn_w,
			height: 1,
		};
		self.save_rect.replace(Some(save_area));
		render_btn(frame, "Save", BTN_SAVE_BG, save_area, is_busy || !is_valid);

		let cancel_area = Rect {
			x: btn_start_x + btn_w + gap,
			y: btn_y,
			width: btn_w,
			height: 1,
		};
		self.cancel_rect.replace(Some(cancel_area));
		render_btn(frame, "Close", BTN_CANCEL_BG, cancel_area, false);
	}

	pub fn handle_key(&mut self, ke: &crossterm::event::KeyEvent) -> ModalAction {
		use crossterm::event::KeyCode;
		let state = self.config_state.read().unwrap_or_else(|p| p.into_inner());
		let is_busy = matches!(*state, ConfigState::Connecting);
		drop(state);
		if is_busy {
			return ModalAction::None;
		}
		if !self.is_valid() && ke.code == KeyCode::Enter {
			return ModalAction::None;
		}

		match ke.code {
			KeyCode::Esc => ModalAction::Close,
			KeyCode::Tab => {
				let fields = if self.requires_auth {
					vec![
						FocusedField::Name,
						FocusedField::Endpoint,
						FocusedField::Username,
						FocusedField::Password,
					]
				} else {
					vec![FocusedField::Name, FocusedField::Endpoint]
				};
				let idx = fields
					.iter()
					.position(|f| f == &self.focused_field)
					.unwrap_or(0);
				self.focused_field = fields[(idx + 1) % fields.len()];
				ModalAction::None
			}
			KeyCode::BackTab => {
				let fields = if self.requires_auth {
					vec![
						FocusedField::Name,
						FocusedField::Endpoint,
						FocusedField::Username,
						FocusedField::Password,
					]
				} else {
					vec![FocusedField::Name, FocusedField::Endpoint]
				};
				let idx = fields
					.iter()
					.position(|f| f == &self.focused_field)
					.unwrap_or(0);
				self.focused_field = fields[(idx + fields.len() - 1) % fields.len()];
				ModalAction::None
			}
			KeyCode::Enter => ModalAction::Save,
			KeyCode::Char(c) => {
				match self.focused_field {
					FocusedField::Name => self.server_name.push(c),
					FocusedField::Endpoint => self.endpoint.push(c),
					FocusedField::Username => self.username.push(c),
					FocusedField::Password => self.password.push(c),
				}
				ModalAction::None
			}
			KeyCode::Backspace => {
				match self.focused_field {
					FocusedField::Name => {
						self.server_name.pop();
					}
					FocusedField::Endpoint => {
						self.endpoint.pop();
					}
					FocusedField::Username => {
						self.username.pop();
					}
					FocusedField::Password => {
						self.password.pop();
					}
				}
				ModalAction::None
			}
			_ => ModalAction::None,
		}
	}

	pub fn handle_mouse(&self, me: &crossterm::event::MouseEvent) -> Option<ModalAction> {
		use crossterm::event::MouseEventKind;
		if let MouseEventKind::Down(_) = me.kind {
			let col = me.column;
			let row = me.row;
			if let Some(r) = *self.save_rect.borrow() {
				if col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height {
					return Some(ModalAction::Save);
				}
			}
			if let Some(r) = *self.cancel_rect.borrow() {
				if col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height {
					return Some(ModalAction::Close);
				}
			}
			if let Some(r) = *self.auth_rect.borrow() {
				if col >= r.x && col < r.x + r.width + 2 && row >= r.y && row < r.y + r.height {
					return Some(ModalAction::ToggleAuth);
				}
			}
		}
		None
	}
}

pub enum ModalAction {
	None,
	Save,
	Close,
	ToggleAuth,
}

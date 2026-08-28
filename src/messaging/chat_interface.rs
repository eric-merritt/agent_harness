// Chat interface — owns the conversation, message list, and orchestrates
// submission → DB save → panel update.  Main delegates to this; it does
// NOT handle events or data flow between components directly.

use std::sync::{Arc, RwLock};

use crossterm::event::MouseEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::style::Style;
use ratatui::widgets::Block;

use crate::database::postgres::Database;
use crate::ui_ux::components::input_bar::InputBar;
use crate::ui_ux::components::messages_panel::MessagesPanel;
use crate::ui_ux::components::submit_button::SubmitButton;

use super::conversation::Conversation;
use super::message::{Message, MessageState, Role};

/// Shared state that the render pass and the async worker both touch.
pub struct ChatState<'a> {
	/// The in-memory conversation (source of truth for the message list).
	pub conversation: Conversation,
	/// TUI panel that renders the messages.
	pub messages_panel: MessagesPanel,
	/// Input bar (shared with main loop for key handling).
	pub input_bar: InputBar,
	/// Submit button widget.
	pub submit_button: SubmitButton<'a>,
	/// Button hover state (not owned by SubmitButton so main can mutate it).
	pub button_hovered: bool,
	/// Button pressed state (toggled on click).
	pub button_pressed: bool,
	/// Auto-release flag: when true, the next render will clear button_pressed.
	/// This ensures the pressed visual is shown for exactly one frame.
	pub button_press_pending: bool,
	/// Last-known button rect for hit-testing (set during render).
	pub last_submit_btn: Option<Rect>,
	/// Last-known messages panel rect for mouse delegation.
	pub last_messages_area: Option<Rect>,
}

pub struct ChatInterface<'a> {
	/// Optional DB handle — if absent, messages stay in-memory only.
	pub(crate) db: Option<Database>,
	/// Shared mutable state behind a std RwLock so the render thread
	/// can read it synchronously inside terminal.draw().
	state: Arc<RwLock<ChatState<'a>>>,
}

impl<'a> ChatInterface<'a> {
	pub fn new(db: Option<Database>) -> Self {
		let conv = Conversation::new();
		Self {
			db,
			state: Arc::new(RwLock::new(ChatState {
				conversation: conv,
				messages_panel: MessagesPanel::new(),
				input_bar: InputBar::new(),
				submit_button: SubmitButton {
					label: "Send",
					is_pressed: false,
					is_hovered: false,
				},
				button_hovered: false,
				button_pressed: false,
				button_press_pending: false,
				last_submit_btn: None,
				last_messages_area: None,
			})),
		}
	}

	/// Return a handle to the input bar so main can feed it key events.
	pub fn input_bar(&self) -> InputBar {
		let guard = self.state.read().unwrap_or_else(|p| p.into_inner());
		guard.input_bar.clone()
	}

	/// Return the submit button widget.
	pub fn submit_button(&self) -> SubmitButton<'static> {
		SubmitButton {
			label: "Send",
			is_pressed: false,
			is_hovered: false,
		}
	}

	/// Set whether the button is currently hovered.
	pub fn set_button_hovered(&self, hovered: bool) {
		let mut state = self.state.write().unwrap_or_else(|p| p.into_inner());
		state.button_hovered = hovered;
	}

	/// Set the button pressed state directly.
	pub fn set_button_pressed(&self, pressed: bool) {
		let mut state = self.state.write().unwrap_or_else(|p| p.into_inner());
		state.button_pressed = pressed;
	}

	/// Mark the button as pressed with auto-release after the next render.
	/// This ensures the pressed visual is shown for exactly one visible frame.
	pub fn set_button_pressed_pending(&self) {
		let mut state = self.state.write().unwrap_or_else(|p| p.into_inner());
		state.button_pressed = true;
		state.button_press_pending = true;
	}

	/// Return the messages panel for rendering.
	pub fn messages_panel(&self) -> MessagesPanel {
		let guard = self.state.read().unwrap_or_else(|p| p.into_inner());
		guard.messages_panel.clone()
	}

	/// Build a message and update in-memory state + panel synchronously.
	/// Returns the conversation id so the caller can persist to DB asynchronously.
	pub fn submit_sync(&self, text: String) -> Option<(uuid::Uuid, String)> {
		if text.is_empty() {
			return None;
		}

		log::info!("submitting user message ({} chars)", text.chars().count());

		// Build the message outside any lock.
		let conv_id = {
			let state = self.state.read().unwrap_or_else(|p| p.into_inner());
			state.conversation.id
		};

		let mut msg = Message::draft(Role::User, conv_id, text.clone());
		msg.state = MessageState::Complete;

		// Update in-memory state immediately so the very next render shows it.
		{
			let mut state = self.state.write().unwrap_or_else(|p| p.into_inner());
			state.conversation.add(msg);

			let snapshot: Vec<Message> = state.conversation.messages.clone();
			state.messages_panel.sync(&snapshot);
		}

		Some((conv_id, text))
	}

	/// Add a pending assistant response (shown while processing).
	pub fn add_pending_response(&self) {
		log::debug!("adding pending response placeholder");
		let conv_id = {
			let state = self.state.read().unwrap_or_else(|p| p.into_inner());
			state.conversation.id
		};
		let mut msg = Message::draft(Role::Agent, conv_id, String::from("Processing..."));
		msg.state = MessageState::Draft;
		let mut state = self.state.write().unwrap_or_else(|p| p.into_inner());
		state.conversation.add(msg);
		let snapshot: Vec<Message> = state.conversation.messages.clone();
		state.messages_panel.sync(&snapshot);
	}

	/// Replace the last assistant message with the given response text.
	pub fn deliver_response(&self, response: String) {
		log::info!(
			"model response received ({} chars)",
			response.chars().count()
		);
		let mut state = self.state.write().unwrap_or_else(|p| p.into_inner());
		// Find the last assistant message and update it
		if let Some(last) = state
			.conversation
			.messages
			.iter_mut()
			.rev()
			.find(|m| m.role == Role::Agent)
		{
			last.content = response;
			last.state = MessageState::Complete;
		}
		let snapshot: Vec<Message> = state.conversation.messages.clone();
		state.messages_panel.sync(&snapshot);
	}

	/// Clone the inner state Arc so async tasks can update the chat.
	pub fn state_handle(&self) -> Arc<RwLock<ChatState<'a>>> {
		Arc::clone(&self.state)
	}

	/// Render the chat-related widgets into the given area.
	/// The layout already splits areas; this just delegates.
	pub fn render(
		&self,
		frame: &mut Frame,
		messages_area: Rect,
		input_area: Rect,
		submit_area: Rect,
		btn_container: Rect,
	) {
		// Paint the button container background (panel bg, not button bg)
		frame.render_widget(
			Block::default().style(Style::default().bg(Color::Rgb(16, 16, 16))),
			btn_container,
		);

		// Snapshot everything we need under one read guard, then drop it
		// before acquiring a write guard (avoids deadlock).
		let (panel, input_bar, label, pressed, hovered, pending) = {
			let state = self.state.read().unwrap_or_else(|p| p.into_inner());
			(
				state.messages_panel.clone(),
				state.input_bar.clone(),
				state.submit_button.label,
				state.button_pressed,
				state.button_hovered,
				state.button_press_pending,
			)
		};

		panel.render(frame, messages_area);
		input_bar.render(frame, input_area);

		let btn = SubmitButton {
			label,
			is_pressed: pressed,
			is_hovered: hovered,
		};
		frame.render_widget(btn, submit_area);

		// Cache rects + auto-clear the pending button press after one render.
		// This ensures the pressed visual is shown for exactly one visible frame.
		{
			let mut state = self.state.write().unwrap_or_else(|p| p.into_inner());
			state.last_submit_btn = Some(submit_area);
			state.last_messages_area = Some(messages_area);
			if pending {
				state.button_pressed = false;
				state.button_press_pending = false;
			}
		}
	}

	/// Take the current input text out of the input bar (empties it).
	/// Returns the text that was in the buffer.
	pub fn take_input(&self) -> String {
		let input_bar = self.input_bar();
		let mut state = input_bar.state.write().unwrap_or_else(|p| p.into_inner());
		state.take()
	}

	/// Scroll the messages panel. Returns true if the key was consumed.
	pub fn scroll_messages(&self, key: &crossterm::event::KeyEvent) -> bool {
		let state = self.state.write().unwrap_or_else(|p| p.into_inner());
		state.messages_panel.handle_key(key)
	}

	/// Handle a mouse event — checks hover/click against the cached button rect.
	/// Returns Some(text) when the button was clicked (mouse-up inside button),
	/// signaling the caller to submit that text.
	pub fn handle_mouse(&self, me: &MouseEvent) -> Option<String> {
		// ── Delegate scroll/mouse events to the messages panel first ──
		{
			let state = self.state.read().unwrap_or_else(|p| p.into_inner());
			if let Some(msg_area) = state.last_messages_area {
				// Clone the panel so we can mutate its scroll state
				let mut panel = state.messages_panel.clone();
				drop(state);

				if panel.handle_mouse(me, msg_area) {
					// Event consumed by the messages panel
					return None;
				}
			} else {
				drop(state);
			}
		}

		// ── Button hit-testing ──
		let state = self.state.read().unwrap_or_else(|p| p.into_inner());
		let Some(btn_rect) = state.last_submit_btn else {
			return None;
		};
		drop(state);

		let in_button = me.column >= btn_rect.x
			&& me.column < btn_rect.x + btn_rect.width
			&& me.row >= btn_rect.y
			&& me.row < btn_rect.y + btn_rect.height;

		match me.kind {
			crossterm::event::MouseEventKind::Moved => {
				self.set_button_hovered(in_button);
				None
			}
			crossterm::event::MouseEventKind::Down(_) => {
				if in_button {
					self.set_button_pressed(true);
				}
				None
			}
			crossterm::event::MouseEventKind::Up(_) => {
				if in_button {
					self.set_button_pressed(false);
					Some(self.take_input())
				} else {
					None
				}
			}
			_ => None,
		}
	}
}

impl<'a> Clone for ChatInterface<'a> {
	fn clone(&self) -> Self {
		Self {
			db: self.db.clone(),
			state: Arc::clone(&self.state),
		}
	}
}

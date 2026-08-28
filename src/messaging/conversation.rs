use super::message::Message;
use uuid::Uuid;

pub struct Conversation {
	pub id: Uuid,
	pub messages: Vec<Message>,
}

impl Conversation {
	pub fn new() -> Self {
		Self {
			id: Uuid::new_v4(),
			messages: Vec::new(),
		}
	}

	pub fn add(&mut self, msg: Message) {
		log::debug!(
			"message added to conversation {}: role={}",
			self.id,
			msg.role
		);
		self.messages.push(msg);
	}
}

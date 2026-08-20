use chrono::Timelike;
use uuid::Uuid;

/// Who produced this message.
#[derive(Clone, Debug, PartialEq)]
pub enum Role {
    User,
    Agent,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::User => write!(f, "[ User ]"),
            Role::Agent => write!(f, "[ Agent ]"),
        }
    }
}

/// Lifecycle state of a message.
#[derive(Clone, Debug, PartialEq)]
pub enum MessageState {
    /// Being streamed in — content is incomplete.
    Draft,
    /// Final — persisted to DB, no more changes.
    Complete,
}

/// A single message in a conversation.
#[derive(Clone, Debug)]
pub struct Message {
    pub id: Uuid,
    pub role: Role,
    pub conv_id: Uuid,
    pub content: String,
    pub state: MessageState,
    /// When this message was created, in UTC.
    /// Displayed in the user's local timezone at render time.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Message {
    /// Create a new draft message (not yet saved).
    pub fn draft(role: Role, conv_id: Uuid, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            role,
            conv_id,
            content: content.into(),
            state: MessageState::Draft,
            created_at: chrono::Utc::now(),
        }
    }

    /// Create a new streaming message (content will be appended to).
    pub fn streaming(role: Role, conv_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            role,
            conv_id,
            content: String::new(),
            state: MessageState::Draft,
            created_at: chrono::Utc::now(),
        }
    }

    /// Append a chunk to the content (for streaming).
    pub fn append(&mut self, chunk: &str) {
        self.content.push_str(chunk);
    }

    /// Mark this message as complete.
    pub fn complete(&mut self) {
        self.state = MessageState::Complete;
    }

    /// Format created_at as a 12-hour local time string, e.g. "9:00pm", "2:30am".
    pub fn local_time(&self) -> String {
        let local: chrono::DateTime<chrono::Local> = self.created_at.with_timezone(&chrono::Local);
        let hour = local.hour();
        let am_pm = if hour >= 12 { "pm" } else { "am" };
        let hour12 = match hour % 12 { 0 => 12, h => h };
        format!("{:02}:{:02}{}", hour12, local.minute(), am_pm)
    }
}

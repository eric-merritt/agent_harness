use ratatui::layout::{Constraint, Layout, Rect};
use special::*;

const WIDE_WIDTH_THRESHOLD: u16 = 120;

const INPUT_MIN_HEIGHT: u16 = 3;

pub struct ChatAreas {
    pub messages: Rect,
    pub input: Rect,
    pub tools: Option<Rect>,
    pub mcp_servers: Option<Rect>,
}

pub fn create_chat_layout(area: Rect) -> ChatAreas {
    if area.width >= WIDE_WIDTH_THRESHOLD {
        let [chat, sidebar] = Layout::horizontal([
            Constraint::Ratio(75, 100),
            Constraint::Ratio(25, 100),
        ])
        .areas(area);

        let [messages, input] = Layout::vertical([
            Constraint::Fill(1),
            Constraint::Min(INPUT_MIN_HEIGHT),
        ])
        .areas(chat);

        let [tools, mcp_servers] = Layout::vertical([
            Constraint::Ratio(50, 100),
            Constraint::Ratio(50, 100),
        ])
        .areas(sidebar);

        ChatAreas {
            messages,
            input,
            tools: Some(tools),
            mcp_servers: Some(mcp_servers),
        }
    } else {
        let [messages, input] = Layout::vertical([
            Constraint::Fill(1),
            Constraint::Min(INPUT_MIN_HEIGHT),
        ])
        .areas(area);

        ChatAreas {
            messages,
            input,
            tools: None,
            mcp_servers: None,
        }
    }
}

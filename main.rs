use ratatui::{
    Frame,
    style::Color,
};

use crossterm::event;

use crate::layout::layout::create_chat_layout;

fn render(frame: &mut Frame) {
    let areas = create_chat_layout(frame.area());

}

fn run(terminal: &mut ratatui::DefaultTerminal) -> std::io::Result<()> {
    loop {
        terminal.draw(render);

    }
}

fn main() -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result = run(&mut terminal);
    ratatui::restore();
    Ok(result?)
}

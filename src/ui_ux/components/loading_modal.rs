// Loading modal — non-closable modal shown during model loading.
// Border style copied from submit_button.rs: 3 magenta tones, all static.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::layout::HorizontalAlignment;
use crate::ui_ux::theme::{Theme, ThemeName};

use crate::progress::LoadingProgress;

const RETRO_THEME: ThemeName = ThemeName::Retro;
// Three magenta tones — same as submit_button.rs.
const MODAL_LIGHT: Color = Color::Rgb(200, 40, 200);  // top/left borders
const MODAL_BG: Color = Color::Rgb(150, 50, 150);     // background fill
const MODAL_DARK: Color = Color::Rgb(100, 20, 100);   // bottom/right borders
const TEXT_FG: Color = Color::Rgb(255, 255, 255);     // white text
const TEAL: Color = Color::Rgb(0, 128, 128);           // teal progress bar bg

const PROGRESS_CHAR: char = '\u{258F}';  // ▏

pub fn render_loading_modal(frame: &mut Frame, progress: &LoadingProgress) {
    let full = frame.area();
    let buf = frame.buffer_mut();

    let w = 50u16.min(full.width.saturating_sub(4));
    let h = 11u16.min(full.height.saturating_sub(4));
    let x = full.x + (full.width.saturating_sub(w)) / 2;
    let y = full.y + (full.height.saturating_sub(h)) / 2;
    let area = Rect { x, y, width: w, height: h };

    let (left, right) = (area.left(), area.right() - 1);
    let (top, bottom) = (area.top(), area.bottom() - 1);

    let mut put = |x: u16, y: u16, ch: char, fg: Color, bg: Color| {
        if let Some(cell) = buf.cell_mut((x, y)) {
            cell.set_char(ch).set_fg(fg).set_bg(bg);
        }
    };

    // Face fill
    for yy in top + 1..bottom {
        for xx in left + 1..right {
            put(xx, yy, ' ', Color::Reset, MODAL_BG);
        }
    }

    // Top edge — light
    for xx in left + 1..right {
        put(xx, top, '▔', MODAL_LIGHT, MODAL_BG);
    }
    // Left edge — light
    for yy in top + 1..bottom {
        put(left, yy, '▎', MODAL_LIGHT, MODAL_BG);
    }
    // Bottom edge — dark
    for xx in left + 1..right {
        put(xx, bottom, '▁', MODAL_DARK, MODAL_BG);
    }
    // Right edge — dark
    for yy in top + 1..bottom {
        put(right, yy, '▊', MODAL_BG, MODAL_DARK);
    }
    // Corners
    put(left, top, '▗', MODAL_BG, MODAL_LIGHT);
    put(right, top, '▖', MODAL_BG, MODAL_LIGHT);
    put(left, bottom, '▝', MODAL_BG, MODAL_DARK);
    put(right, bottom, '▘', MODAL_BG, MODAL_DARK);

    // Content area inside borders
    let inner_x = left + 1;
    let inner_y = top + 1;
    let inner_w = w.saturating_sub(2);
    let inner_h = h.saturating_sub(2);

    let status = progress.get_status();
    let pct = progress.get_pct();

    // Build the 3 content lines + gaps
    let header = Span::styled(
        "MODEL LOADING. PLEASE WAIT.",
        Style::default().fg(TEXT_FG).bg(MODAL_BG).add_modifier(Modifier::BOLD).add_modifier(Modifier::UNDERLINED),
    );

    let status_span = Span::styled(
        format!(" {} ", status),
        Style::default().fg(TEXT_FG).bg(MODAL_BG).add_modifier(Modifier::BOLD),
    );

    let bar_width = inner_w as usize;
    let filled = (bar_width * pct as usize) / 100;
    let bar_text = std::iter::repeat(PROGRESS_CHAR).take(filled).collect::<String>();
    let bar = Span::styled(
        bar_text,
        Style::default().fg(MODAL_BG).bg(TEAL),
    );

    let pct_span = Span::styled(
        format!(" {:>3}% ", pct),
        Style::default().fg(TEXT_FG).bg(MODAL_BG).add_modifier(Modifier::BOLD),
    );

    // Lines: header, gap, status, gap, progress bar, percentage
    let lines = vec![
        Line::from(header),
        Line::raw(""),
        Line::from(status_span),
        Line::raw(""),
        Line::from(bar),
        Line::from(pct_span),
    ];
    let content = Text::from(lines);

    let inner_border = Theme::retro().bg;
    // Outer paragraph — terminal BG border in default style, everything centered
    let content_block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(inner_border));
    let content_para = Paragraph::new(content)
        .block(content_block)
        .alignment(HorizontalAlignment::Center);

    let content_area = Rect::new(inner_x, inner_y, inner_w, inner_h);
    frame.render_widget(content_para, content_area);
}

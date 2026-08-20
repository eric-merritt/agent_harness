// Application driver — owns foundational TUI state, the draw loop,
// and event routing.  main.rs only declares modules, initializes the DB,
// and hands off to this struct.

use std::sync::{Arc, atomic::AtomicBool, atomic::Ordering, RwLock};

use crossterm::event::{
    Event, KeyCode, KeyEventKind, KeyModifiers,
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
};
use ratatui::style::{Color, Style};
use ratatui::widgets::Block;

use crate::database::postgres::Database;
use crate::messaging::chat_interface::ChatInterface;
use crate::messaging::layout::DefaultLayout;
use crate::messaging::mcp_config::{McpConfig, SavedMcpServer};
use crate::models::server::ModelServer;
use crate::progress::LoadingProgress;
use crate::ui_ux::components::mcp_panel::{McpPanel, McpToolNode};
use crate::ui_ux::components::mcp_modal::{McpModal, ModalAction, ConfigState};
use crate::ui_ux::components::tools_panel::ToolsPanel;
use crate::ui_ux::components::loading_modal::render_loading_modal;

/// The application driver. Holds all top-level UI state and runs the loop.
pub struct App {
    chat: ChatInterface<'static>,
    tools_panel: ToolsPanel,
    mcp_panel: McpPanel,
    dirty: Arc<AtomicBool>,
    /// When true, show the MCP server configuration modal.
    show_mcp_modal: bool,
    mcp_modal: McpModal,
    /// Server info to add to MCP panel after successful connection test.
    pending_mcp_server: Arc<RwLock<Option<(String, String, String, usize)>>>,
    /// Compressed model server (if a model was detected on startup).
    model_server: Option<Arc<ModelServer>>,
    /// Last-known layout areas (set during render, used for mouse hit-testing).
    last_areas: Option<DefaultLayout>,
    /// Model loading progress tracker — when Some, the loading modal is shown.
    loading_progress: Option<LoadingProgress>,
    /// When loading started — used to enforce minimum modal display time.
    loading_started: Option<std::time::Instant>,
    /// Shared slot for the model server being loaded in background.
    loading_server: Arc<RwLock<Option<Arc<ModelServer>>>>,
}

impl App {
    /// Instantiate the App from the optional database pool and model path.
    /// If a model path is given, spawns a background thread to load the model
    /// and shows a non-closable loading modal until it completes.
    pub fn new(db: Option<Database>, model_path: Option<std::path::PathBuf>) -> Self {
        let chat = ChatInterface::new(db);
        let tools_panel = ToolsPanel::new();
        let mut mcp_panel = McpPanel::new();
        let dirty = Arc::new(AtomicBool::new(true));
        let mcp_modal = McpModal::new();
        let loading_server: Arc<RwLock<Option<Arc<ModelServer>>>> = Arc::new(RwLock::new(None));

        // Load saved MCP servers from config
        let config = McpConfig::load();
        for server in &config.servers {
            mcp_panel.add_card(
                server.name.clone(),
                server.transport.clone(),
                server.endpoint.clone(),
                true,
                0,
            );
        }

        // If a model path was given, start background loading with progress tracking
        let loading_started = model_path.as_ref().map(|_| std::time::Instant::now());
        let loading_progress = model_path.map(|path| {
            let progress = LoadingProgress::new();
            let loading_server = Arc::clone(&loading_server);
            let dirty = Arc::clone(&dirty);
            progress.set(0, "Starting model load...");
            let p = progress.clone();

            std::thread::spawn(move || {
                log::info!("Background model loading from {}", path.display());
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    ModelServer::open_with_progress(&path, Some(&p))
                }));
                match result {
                    Ok(Ok(server)) => {
                        p.finish();
                        *loading_server.write().unwrap() = Some(Arc::new(server));
                        dirty.store(true, Ordering::SeqCst);
                        log::info!("Background model loading complete");
                    }
                    Ok(Err(e)) => {
                        p.fail(&format!("Error: {}", e));
                        dirty.store(true, Ordering::SeqCst);
                        log::error!("Background model loading failed: {}", e);
                    }
                    Err(panic_info) => {
                        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_info.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "unknown panic".to_string()
                        };
                        p.fail(&format!("PANIC: {}", msg));
                        dirty.store(true, Ordering::SeqCst);
                        log::error!("Background model loading PANICKED: {}", msg);
                    }
                }
            });

            progress
        });

        Self {
            chat,
            tools_panel,
            mcp_panel,
            dirty,
            show_mcp_modal: false,
            mcp_modal,
            pending_mcp_server: Arc::new(RwLock::new(None)),
            model_server: None,
            last_areas: None,
            loading_progress,
            loading_started,
            loading_server,
        }
    }

    /// Start an async connection test — tries each transport in priority order.
    fn start_connection_test(&self) {
        let Some((name, endpoint, requires_auth, username, password)) = self.mcp_modal.connect() else {
            return;
        };
        let state = Arc::clone(&self.mcp_modal.config_state);
        let dirty = Arc::clone(&self.dirty);
        let pending = Arc::clone(&self.pending_mcp_server);

        *state.write().unwrap_or_else(|p| p.into_inner()) = ConfigState::Connecting;
        dirty.store(true, Ordering::SeqCst);

        tokio::spawn(async move {
            // Try transports in priority order: Streamable HTTP → SSE → HTTP → gRPC → Stdio
            let result: Result<String, String> = async {
                // Normalize endpoint — try with the raw address for HTTP transports
                let raw = endpoint.trim();
                if raw.is_empty() {
                    return Err("Endpoint is empty".to_string());
                }

                let client = reqwest::Client::new();

                // --- Streamable HTTP (POST with MCP init) ---
                let init_payload = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "agent_harness", "version": "0.1.0"}
                    }
                });
                let mut req = client.post(raw)
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json, text/event-stream")
                    .json(&init_payload)
                    .timeout(std::time::Duration::from_secs(5));
                if requires_auth && !username.is_empty() {
                    req = req.basic_auth(&username, Some(&password));
                }
                if let Ok(resp) = req.send().await {
                    if resp.status().is_success() {
                        let ct = resp.headers().get("content-type")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("")
                            .to_string();
                        if let Ok(body) = resp.text().await {
                            if body.contains("jsonrpc") || ct.contains("text/event-stream") || ct.contains("application/json") {
                                return Ok("Streamable HTTP".to_string());
                            }
                        }
                    }
                }

                // --- SSE (GET with Accept text/event-stream) ---
                let mut req = client.get(raw)
                    .header("Accept", "text/event-stream")
                    .timeout(std::time::Duration::from_secs(5));
                if requires_auth && !username.is_empty() {
                    req = req.basic_auth(&username, Some(&password));
                }
                if let Ok(resp) = req.send().await {
                    if resp.status().is_success() {
                        let ct = resp.headers().get("content-type")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("")
                            .to_string();
                        if ct.contains("text/event-stream") {
                            return Ok("SSE".to_string());
                        }
                    }
                }

                // --- Plain HTTP (simple GET) ---
                let mut req = client.get(raw)
                    .timeout(std::time::Duration::from_secs(5));
                if requires_auth && !username.is_empty() {
                    req = req.basic_auth(&username, Some(&password));
                }
                if let Ok(resp) = req.send().await {
                    if resp.status().is_success() {
                        return Ok("HTTP".to_string());
                    }
                }

                // --- gRPC (TCP connect) ---
                let grpc_addr = raw
                    .strip_prefix("grpc://").or_else(|| raw.strip_prefix("http://"))
                    .or_else(|| raw.strip_prefix("https://"))
                    .unwrap_or(raw)
                    .trim_end_matches('/');
                if tokio::net::TcpStream::connect(grpc_addr).await.is_ok() {
                    return Ok("gRPC".to_string());
                }

                // --- Stdio (command exists) ---
                let cmd = raw.split_whitespace().next().unwrap_or("");
                if !cmd.is_empty() && (std::process::Command::new("which").arg(cmd).output().is_ok()
                    || std::path::Path::new(cmd).exists()) {
                    return Ok("Stdio".to_string());
                }

                Err("No transport succeeded".to_string())
            }.await;

            let mut state = state.write().unwrap_or_else(|p| p.into_inner());
            *state = match result {
                Ok(transport) => {
                    *pending.write().unwrap_or_else(|p| p.into_inner()) = Some((name, transport, endpoint.clone(), 0));
                    ConfigState::Connected
                }
                Err(msg) => ConfigState::Failed(msg),
            };
            dirty.store(true, Ordering::SeqCst);
        });
    }
    pub fn run(mut self) {
        log::info!("app starting");
        let mut terminal = ratatui::init();
        let mut stdout = std::io::stdout();
        if let Err(e) = crossterm::execute!(
            stdout,
            EnableMouseCapture,
            EnableBracketedPaste,
        ) {
            log::warn!("failed to enable mouse/paste capture: {}", e);
        }

        self.inner_loop(&mut terminal);

        // Restore the terminal, then explicitly disable features we enabled.
        // Order matters: restore first (leaves alt-screen, re-enables canonical mode,
        // shows cursor), then disable mouse/paste so the shell doesn't interpret
        // subsequent input as mouse events or paste sequences.
        log::info!("app shutting down");
        ratatui::restore();
        drop(terminal);
        if let Err(e) = crossterm::execute!(
            std::io::stdout(),
            DisableMouseCapture,
            DisableBracketedPaste,
        ) {
            log::warn!("failed to disable mouse/paste capture: {}", e);
        }
    }

    // ────────────────────────── inner loop ──────────────────────────

    fn inner_loop(&mut self, terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>) {
        let mut last_redraw = std::time::Instant::now();
        let redraw_interval = std::time::Duration::from_millis(166);
        let mut running = true;

        loop {
            if !running {
                break;
            }

            // ── Draw ─
            let needs_periodic_redraw = last_redraw.elapsed() >= redraw_interval;
            let needs_redraw = self.dirty.swap(false, Ordering::SeqCst) || needs_periodic_redraw;
            if needs_periodic_redraw {
                last_redraw = std::time::Instant::now();
                let input_bar = self.chat.input_bar();
                if let Ok(mut state) = input_bar.state.write() {
                    state.cursor_visible = !state.cursor_visible;
                }
            }

            if needs_redraw {
                if let Err(e) = terminal.draw(|frame| self.render_mut(frame)) {
                    log::error!("terminal draw failed: {}", e);
                }
            }

            // ── Input ─
            if crossterm::event::poll(std::time::Duration::from_millis(100)).unwrap_or(false) {
                let event = match crossterm::event::read() {
                    Ok(ev) => ev,
                    Err(e) => {
                        log::error!("failed to read terminal event: {}", e);
                        continue;
                    }
                };
                if !self.handle_event(event) {
                    running = false;
                }
            }
        }
    }

    // ────────────────────────── render ──────────────────────────────

    fn render_mut(&mut self, frame: &mut ratatui::Frame) {
        // Auto-close modal on successful connection + add server to panel
        if self.show_mcp_modal && self.mcp_modal.is_connected() {
            if let Ok(mut pending) = self.pending_mcp_server.write() {
                if let Some((name, transport, endpoint, tools)) = pending.take() {
                    self.mcp_panel.add_card(name.clone(), transport.clone(), endpoint.clone(), true, tools);
                    let mut config = McpConfig::load();
                    config.add_server(SavedMcpServer { name, transport, endpoint });
                }
            }
            self.show_mcp_modal = false;
            self.mcp_modal.reset();
            self.dirty.store(true, Ordering::SeqCst);
        } else if let Ok(pending) = self.pending_mcp_server.read() {
            if pending.is_some() {
                drop(pending);
                if let Ok(mut pending) = self.pending_mcp_server.write() {
                    if let Some((name, transport, endpoint, tools)) = pending.take() {
                        self.mcp_panel.add_card(name.clone(), transport.clone(), endpoint.clone(), true, tools);
                        let mut config = McpConfig::load();
                        config.add_server(SavedMcpServer { name, transport, endpoint });
                        self.dirty.store(true, Ordering::SeqCst);
                    }
                }
            }
        }

        // Black outer background
        frame.render_widget(
            Block::default().style(Style::default().bg(Color::Black)),
            frame.area(),
        );

        let areas = DefaultLayout::resolve(frame.area());
        self.last_areas = Some(areas);

        // ── Chat widgets (messages, input, submit) ──
        self.chat.render(
            frame,
            areas.messages,
            areas.user_input,
            areas.submit_btn,
            areas.btn_container_bg,
        );

        // ── Tools panel ──
        if !areas.tools.is_empty() && areas.tools.width >= 3 {
            self.tools_panel.render(frame, areas.tools);
        }

        // ── MCP panel ──
        if !areas.mcp.is_empty() && areas.mcp.width >= 3 {
            self.mcp_panel.render(frame, areas.mcp);
        }

        // ── MCP modal (overlay) ──
        if self.show_mcp_modal {
            self.mcp_modal.render(frame);
        }

        // ── Loading modal (overlay, non-closable) ──
        // Stays on display until the model is FULLY loaded — all tensors decompressed.
        // This ensures inference works instantly when the modal disappears.
        if let Some(ref lp) = self.loading_progress {
            if lp.is_done() {
                // Model fully loaded — transition server and dismiss modal
                if let Ok(mut slot) = self.loading_server.write() {
                    if let Some(server) = slot.take() {
                        self.model_server = Some(server);
                        log::info!("Model server transitioned to active — all tensors cached");
                    }
                }
                self.loading_progress = None;
                self.loading_started = None;
                self.dirty.store(true, Ordering::SeqCst);
            } else {
                // Still loading — keep modal ON DISPLAY
                render_loading_modal(frame, lp);
                self.dirty.store(true, Ordering::SeqCst);
            }
        }
    }

    // ────────────────────────── events ──────────────────────────────

    /// Returns `false` to signal the app should exit.
    fn handle_event(&mut self, event: Event) -> bool {
        match event {
            Event::Key(ke) => {
                match ke.kind {
                    KeyEventKind::Press | KeyEventKind::Repeat => {
                        // Loading modal is active — block ALL input (non-closable by user).
                        // Only Ctrl+C is allowed to exit the app during loading.
                        if self.loading_progress.is_some() {
                            if ke.code == KeyCode::Char('c')
                                && ke.modifiers.contains(KeyModifiers::CONTROL)
                            {
                                log::info!("exiting on Ctrl+C during model load");
                                return false;
                            }
                            return true;
                        }

                        // Modal active — handle all keys through modal
                        if self.show_mcp_modal {
                            match self.mcp_modal.handle_key(&ke) {
                                ModalAction::Save => {
                                    self.start_connection_test();
                                }
                                ModalAction::Close => {
                                    self.show_mcp_modal = false;
                                    self.mcp_modal.reset();
                                }
                                ModalAction::ToggleAuth => {
                                    self.mcp_modal.requires_auth = !self.mcp_modal.requires_auth;
                                }
                                ModalAction::None => {}
                            }
                            self.dirty.store(true, Ordering::SeqCst);
                            return true;
                        }

                        // Ctrl+C → exit
                        if ke.code == KeyCode::Char('c')
                            && ke.modifiers.contains(KeyModifiers::CONTROL)
                        {
                            log::info!("exiting on Ctrl+C");
                            self.dirty.store(true, Ordering::SeqCst);
                            return false;
                        }

                        // Scroll keys — only when input bar is empty or cursor is at start
                        let input_bar = self.chat.input_bar();
                        let scroll_allowed = {
                            let s = input_bar.state.read().unwrap_or_else(|p| p.into_inner());
                            s.buffer.is_empty() || s.cursor_pos == 0
                        };

                        if scroll_allowed && self.chat.scroll_messages(&ke) {
                            self.dirty.store(true, Ordering::SeqCst);
                        } else {
                            let mut state = input_bar.state.write().unwrap_or_else(|p| p.into_inner());
                            let handled = state.handle_key(&ke);

                            if handled {
                                if ke.code == KeyCode::Enter {
                                    // Mark button as pressed-pending — the render pass will
                                    // show the pressed visual for one frame, then auto-release it.
                                    self.chat.set_button_pressed_pending();
                                    let text = state.take();
                                    if !text.is_empty() {
                                        log::info!("user submitted message via keyboard");
                                        let chat_ref = self.chat.clone();
                                        if let Some((conv_id, content)) = chat_ref.submit_sync(text) {
                                            let content_for_model = content.clone();
                                            // DB persistence (fire-and-forget)
                                            if let Some(db_pool) = chat_ref.db.clone() {
                                                tokio::spawn(async move {
                                                    if let Err(e) = db_pool.save_message(conv_id, "user", &content).await {
                                                        log::error!("DB save failed: {}", e);
                                                    }
                                                });
                                            }
                                            // Route to local model if loaded
                                            if let Some(ref server) = self.model_server {
                                                log::debug!("routing message to local model");
                                                chat_ref.add_pending_response();
                                                let server = Arc::clone(server);
                                                let chat = self.chat.clone();
                                                tokio::task::spawn_blocking(move || {
                                                    let response = server.process_message(&content_for_model);
                                                    chat.deliver_response(response);
                                                });
                                            }
                                        }
                                    }
                                }
                                self.dirty.store(true, Ordering::SeqCst);
                            }
                        }
                        true
                    }
                    KeyEventKind::Release => {
                        // Release button visual on Enter key release
                        if ke.code == KeyCode::Enter {
                            self.chat.set_button_pressed(false);
                            self.dirty.store(true, Ordering::SeqCst);
                        }
                        true
                    }
                }
            }
            Event::Mouse(me) => {
                // Determine which panel the mouse is over
                let in_mcp = self.last_areas.map(|a| {
                    let m = a.mcp;
                    me.column >= m.x && me.column < m.x + m.width
                        && me.row >= m.y && me.row < m.y + m.height
                }).unwrap_or(false);
                let in_chat = self.last_areas.map(|a| {
                    let m = a.messages;
                    me.column >= m.x && me.column < m.x + m.width
                        && me.row >= m.y && me.row < m.y + m.height
                }).unwrap_or(false);

                // MCP modal mouse handling (highest priority)
                if self.show_mcp_modal {
                    if let Some(action) = self.mcp_modal.handle_mouse(&me) {
                        match action {
                            ModalAction::Save => {
                                self.start_connection_test();
                            }
                            ModalAction::Close => {
                                self.show_mcp_modal = false;
                                    self.mcp_modal.reset();
                            }
                            ModalAction::ToggleAuth => {
                                self.mcp_modal.requires_auth = !self.mcp_modal.requires_auth;
                            }
                            ModalAction::None => {}
                        }
                        self.dirty.store(true, Ordering::SeqCst);
                        return true;
                    }
                }

                // Scroll events — route to whichever panel the mouse is over
                if me.kind == crossterm::event::MouseEventKind::ScrollUp {
                    if in_mcp {
                        self.mcp_panel.scroll_up();
                        self.dirty.store(true, Ordering::SeqCst);
                        return true;
                    }
                    // Chat scroll is handled by chat.handle_mouse below
                }
                if me.kind == crossterm::event::MouseEventKind::ScrollDown {
                    if in_mcp {
                        self.mcp_panel.scroll_down();
                        self.dirty.store(true, Ordering::SeqCst);
                        return true;
                    }
                    // Chat scroll is handled by chat.handle_mouse below
                }

                // Non-scroll mouse events in the sidebar (MCP/tools)
                if in_mcp || !in_chat {
                    // Check MCP "+" button click
                    if self.mcp_panel.handle_mouse(&me) {
                        self.mcp_modal.open();
                        self.show_mcp_modal = true;
                        self.dirty.store(true, Ordering::SeqCst);
                        return true;
                    }
                    // Check MCP group click — toggle group expansion
                    if let Some(group_name) = self.mcp_panel.handle_group_click(&me) {
                        self.mcp_panel.toggle_group(&group_name);
                        self.dirty.store(true, Ordering::SeqCst);
                        return true;
                    }
                    // Check MCP card click — toggle expand and fetch tools
                    if let Some(idx) = self.mcp_panel.handle_card_click(&me) {
                    self.mcp_panel.toggle_card(idx);
                    if self.mcp_panel.cards[idx].expanded && self.mcp_panel.cards[idx].connected {
                        let endpoint = self.mcp_panel.cards[idx].endpoint.clone();
                        let tree = Arc::clone(&self.mcp_panel.tool_tree);
                        let dirty = Arc::clone(&self.dirty);
                        // Clear tree and show "Loading..."
                        *tree.write().unwrap_or_else(|p| p.into_inner()) = None;
                        dirty.store(true, Ordering::SeqCst);
                        tokio::spawn(async move {
                            let client = reqwest::Client::new();
                            let payload = serde_json::json!({
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "tools/list",
                                "params": {}
                            });
                            let result = client.post(&endpoint)
                                .header("Content-Type", "application/json")
                                .header("Accept", "application/json, text/event-stream")
                                .json(&payload)
                                .timeout(std::time::Duration::from_secs(10))
                                .send().await;
                            let nodes = match result {
                                Ok(resp) if resp.status().is_success() => {
                                    match resp.json::<serde_json::Value>().await {
                                        Ok(json) => parse_tool_tree(&json),
                                        Err(e) => {
                                            log::warn!("failed to parse tools/list response: {}", e);
                                            Vec::new()
                                        }
                                    }
                                }
                                Ok(resp) => {
                                    log::warn!("tools/list returned HTTP {}", resp.status());
                                    Vec::new()
                                }
                                Err(e) => {
                                    log::warn!("tools/list request failed: {}", e);
                                    Vec::new()
                                }
                            };
                            *tree.write().unwrap_or_else(|p| p.into_inner()) = Some(nodes);
                            dirty.store(true, Ordering::SeqCst);
                        });
                    }
                    self.dirty.store(true, Ordering::SeqCst);
                    return true;
                    }
                }
                if let Some(text) = self.chat.handle_mouse(&me) {
                    if !text.is_empty() {
                        log::info!("user submitted message via mouse click");
                        let chat_ref = self.chat.clone();
                        if let Some((conv_id, content)) = chat_ref.submit_sync(text) {
                            let content_for_model = content.clone();
                            if let Some(db_pool) = chat_ref.db.clone() {
                                tokio::spawn(async move {
                                    if let Err(e) = db_pool.save_message(conv_id, "user", &content).await {
                                        log::error!("DB save failed: {}", e);
                                    }
                                });
                            }
                            // Route to local model if loaded
                            if let Some(ref server) = self.model_server {
                                log::debug!("routing message to local model");
                                chat_ref.add_pending_response();
                                let server = Arc::clone(server);
                                let chat = self.chat.clone();
                                tokio::task::spawn_blocking(move || {
                                    let response = server.process_message(&content_for_model);
                                    chat.deliver_response(response);
                                });
                            }
                        }
                    }
                }
                self.dirty.store(true, Ordering::SeqCst);
                true
            }
            Event::Resize(_w, _h) => {
                self.dirty.store(true, Ordering::SeqCst);
                true
            }
            // Bracketed paste — insert text directly, no Enter/submit triggered.
            Event::Paste(text) => {
                let input_bar = self.chat.input_bar();
                let mut state = input_bar.state.write().unwrap_or_else(|p| p.into_inner());
                let clean: String = text.chars().filter(|c| *c != '\n' && *c != '\r').collect();
                if !clean.is_empty() {
                    let before: String = state.buffer[..state.cursor_pos].chars().collect();
                    let after: String = state.buffer[state.cursor_pos..].chars().collect();
                    state.buffer = format!("{}{}{}", before, clean, after);
                    state.cursor_pos = before.chars().count() + clean.chars().count();
                }
                self.dirty.store(true, Ordering::SeqCst);
                true
            }
            _ => true,
        }
    }
}

/// Parse MCP tools/list JSON response into a tree of McpToolNode.
/// Tools are grouped using the built-in tool group map (from Python tool files).
/// Unknown tools go into an "ungrouped" bucket.
fn parse_tool_tree(json: &serde_json::Value) -> Vec<McpToolNode> {
    let tools_arr = json
        .get("result")
        .and_then(|r| r.get("tools"))
        .and_then(|t| t.as_array());

    let mut root_nodes: Vec<McpToolNode> = Vec::new();

    let tools = match tools_arr {
        Some(arr) => arr,
        None => return root_nodes,
    };

    let group_map = McpConfig::load().tool_to_group_map();

    for tool in tools {
        let name = tool.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
        let desc = tool.get("description").and_then(|d| d.as_str()).unwrap_or("");

        let group_name = group_map.get(name).cloned().unwrap_or_else(|| "ungrouped".to_string());

        let group_idx = root_nodes.iter().position(|n| n.name == group_name);
        match group_idx {
            Some(idx) => {
                root_nodes[idx].children.push(McpToolNode {
                    name: name.to_string(),
                    description: desc.to_string(),
                    children: Vec::new(),
                    is_leaf: true,
                });
            }
            None => {
                root_nodes.push(McpToolNode {
                    name: group_name.clone(),
                    description: String::new(),
                    children: vec![McpToolNode {
                        name: name.to_string(),
                        description: desc.to_string(),
                        children: Vec::new(),
                        is_leaf: true,
                    }],
                    is_leaf: false,
                });
            }
        }
    }

    root_nodes
}

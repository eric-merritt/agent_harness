// MCP server configuration persistence — saves/loads servers and tool groups to JSON.

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedMcpServer {
    pub name: String,
    pub transport: String,
    pub endpoint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolGroupDef {
    pub group_name: String,
    pub tools: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct McpConfig {
    pub servers: Vec<SavedMcpServer>,
    pub tool_groups: Vec<ToolGroupDef>,
}

impl McpConfig {
    fn config_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".config").join("agent_harness").join("mcp_servers.json")
    }

    /// Load config from disk. Seeds builtin tool groups on first run.
    pub fn load() -> Self {
        let path = Self::config_path();
        log::debug!("loading MCP config from {:?}", path);
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                let mut cfg: Self = match serde_json::from_str(&content) {
                    Ok(c) => c,
                    Err(e) => {
                        log::warn!("failed to parse MCP config JSON, using defaults: {}", e);
                        Self::default()
                    }
                };
                if cfg.tool_groups.is_empty() {
                    cfg.tool_groups = builtin_tool_groups();
                    cfg.save();
                }
                log::info!("loaded {} MCP servers from config", cfg.servers.len());
                cfg
            }
            Err(e) => {
                log::info!("MCP config not found, creating default: {}", e);
                let cfg = Self { servers: Vec::new(), tool_groups: builtin_tool_groups() };
                cfg.save();
                cfg
            }
        }
    }

    pub fn save(&self) {
        let path = Self::config_path();
        log::debug!("saving MCP config to {:?}", path);
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                log::error!("failed to create config dir {:?}: {}", parent, e);
            }
        }
        match serde_json::to_string_pretty(self) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    log::error!("failed to write MCP config to {:?}: {}", path, e);
                }
            }
            Err(e) => {
                log::error!("failed to serialize MCP config: {}", e);
            }
        }
    }

    pub fn add_server(&mut self, server: SavedMcpServer) {
        if !self.servers.iter().any(|s| s.endpoint == server.endpoint && s.name == server.name) {
            self.servers.push(server);
            self.save();
        }
    }

    pub fn tool_to_group_map(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        for group in &self.tool_groups {
            for tool in &group.tools {
                map.insert(tool.clone(), group.group_name.clone());
            }
        }
        map
    }
}

fn builtin_tool_groups() -> Vec<ToolGroupDef> {
    vec![
        ToolGroupDef { group_name: "filesystem".into(), tools: vec![
            "ReadFile".into(), "FileInfo".into(), "SummarizeFileContent".into(),
            "WriteFile".into(), "SearchFileForPattern".into(), "FindFnDefInFile".into(),
            "ListFilesInDirectory".into(), "MoveFileOrDirectory".into(), "FindFileDirByName".into(),
        ]},
        ToolGroupDef { group_name: "web".into(), tools: vec![
            "WebSetCookies".into(), "WebSetLocalStorage".into(), "CurrentCookiesForURL".into(),
            "CurrentCookies".into(), "WebSearchGeneral".into(), "FetchAndExtractContent".into(),
            "FetchFromWebsite".into(), "ReadSummaryRef".into(), "WebsiteFindDirectDownloadLink".into(),
            "WebFileDownload".into(), "WebFileDownloadStatus".into(), "GetAllowedCrawlPaths".into(),
            "WebsiteQuery".into(), "BrowserClick".into(), "GenerateWebsiteStructure".into(),
        ]},
        ToolGroupDef { group_name: "exploit".into(), tools: vec![
            "ExploitSqlInjection".into(), "ExploitXss".into(), "ExploitSsrf".into(),
            "ExploitCommandInjection".into(), "ExploitPathTraversal".into(), "ExploitRce".into(),
            "ScanTargetForVulns".into(), "GenExploitPayload".into(),
            "ExploitIpCameras".into(), "ScanIpCameras".into(), "FindIpCameraIpRanges".into(),
        ]},
        ToolGroupDef { group_name: "bug_bounty".into(), tools: vec![
            "GetHackerOnePrograms".into(), "GetHackerOneDisclosures".into(), "GetHackerOneCompanyProgram".into(),
            "GetBugcrowdPrograms".into(), "GetBugcrowdDisclosures".into(),
            "GetIntigritiPrograms".into(), "GetYesWeHackPrograms".into(), "GetSynackPrograms".into(),
            "SearchBugBountyPrograms".into(), "GetVulnTypesAndMethodology".into(),
        ]},
        ToolGroupDef { group_name: "torrent".into(), tools: vec![
            "TorrentSearch".into(), "TorrentPlugins".into(), "ToggleTorrentPlugin".into(),
            "AddTorrent".into(), "ActiveTorrents".into(), "DownloadTorrent".into(),
        ]},
        ToolGroupDef { group_name: "onlyfans".into(), tools: vec![
            "ScrollOnlyfansConversations".into(), "ScrollOnlyfansMessages".into(),
            "OnlyfansDownload".into(), "OnlyfansExtractCurrentConversation".into(),
            "OnlyfansExtractAllConversations".into(),
        ]},
        ToolGroupDef { group_name: "presentation".into(), tools: vec![
            "DisplayImage".into(), "EmbedVideo".into(), "RenderAsText".into(),
            "PresentGalleryForDownload".into(), "RenderAsMarkdown".into(),
        ]},
        ToolGroupDef { group_name: "accounting".into(), tools: vec![
            "InitializeLedger".into(), "AccountCreate".into(), "AccountsList".into(),
            "AccountBalance".into(), "UpdateAccount".into(), "NewTransaction".into(),
            "SearchTransactions".into(), "VoidTransaction".into(), "AccountDetails".into(),
            "NewAccountingItem".into(), "ReceiveInventory".into(), "ListAccountingItems".into(),
            "RemoveItemFromInventory".into(), "JournalizeSalesTransaction".into(),
            "GetValueOfAccount".into(), "GenerateFinancialStatement".into(), "AccountClose".into(),
        ]},
        ToolGroupDef { group_name: "browser_session".into(), tools: vec![
            "OpenUrlInBrowser".into(), "WebsiteFormFill".into(), "WebsiteLogin".into(),
        ]},
        ToolGroupDef { group_name: "recorder".into(), tools: vec![
            "BrowserInteractionStartRecording".into(), "BrowserInteractionStopRecording".into(),
            "SaveRecordedBrowserInteraction".into(),
        ]},
        ToolGroupDef { group_name: "tasklist".into(), tools: vec![
            "ReferenceTasklist".into(), "TasklistAppend".into(), "MarkTaskComplete".into(),
        ]},
        ToolGroupDef { group_name: "khan".into(), tools: vec![
            "KhanAcademyCourses".into(), "KhanAcademyCourseUnits".into(), "KhanAcademyLessons".into(),
        ]},
        ToolGroupDef { group_name: "home_automation.cync".into(), tools: vec![
            "AuthLoginCyncByGE".into(), "SaveGECyncAuthResponse".into(), "CheckCyncAuthStatus".into(),
            "ListCyncDevices".into(), "SetCyncDevicePower".into(), "SetCyncLightBrightness".into(),
        ]},
        ToolGroupDef { group_name: "home_automation.lifx".into(), tools: vec![
            "DimLIFXLight".into(), "ChangeColorLIFXLight".into(), "SetLIFXMode".into(),
            "ListLIFXLights".into(), "ListLIFXDevices".into(),
        ]},
        ToolGroupDef { group_name: "context7".into(), tools: vec![
            "Context7ResolveLibraryId".into(), "Context7QueryDocs".into(),
        ]},
        ToolGroupDef { group_name: "ecommerce".into(), tools: vec![
            "EcommerceSearch".into(), "EnrichEcommerceData".into(),
        ]},
        ToolGroupDef { group_name: "jobs".into(), tools: vec![
            "SearchForJobs".into(), "FetchJobBoard".into(),
        ]},
        ToolGroupDef { group_name: "mcp".into(), tools: vec![
            "MCPInitializeConnection".into(), "MCPCallTool".into(),
        ]},
        ToolGroupDef { group_name: "native".into(), tools: vec![
            "ParametersForToolByName".into(), "ListAvailableTools".into(),
        ]},
        ToolGroupDef { group_name: "vector_store".into(), tools: vec![
            "VectorDatabaseKnowledgeStore".into(), "VectorDatabaseKnowledgeSearch".into(),
        ]},
        ToolGroupDef { group_name: "vision".into(), tools: vec![
            "DescribeImage".into(), "SortImagesByQuality".into(),
        ]},
        ToolGroupDef { group_name: "agent_smith".into(), tools: vec![
            "LoadPresetAgent".into(), "CreateCustomAgent".into(),
        ]},
        ToolGroupDef { group_name: "cli".into(), tools: vec![
            "Bash".into(),
        ]},
        ToolGroupDef { group_name: "taskgraph".into(), tools: vec![
            "SynchronizeTaskGraph".into(),
        ]},
    ]
}

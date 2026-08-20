// Rust implementations of filesystem tools from Python:
//   ReadFile, FileInfo, SummarizeFileContent, WriteFile,
//   SearchFileForPattern, FindFnDefInFile, ListFilesInDirectory,
//   MoveFileOrDirectory, FindFileDirByName

use std::path::{Path, PathBuf};
use std::process::Command;
use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use super::tool::{Tool, ToolContext, ToolResult};

/// Max lines returned per read call.
const READ_PAGE: usize = 200;

/// Resolve a path: expand ~, place bare names into workspace.
fn resolve(path: &str) -> PathBuf {
    let expanded = shellexpand::tilde(path).into_owned();
    let p = PathBuf::from(expanded);
    if p.is_absolute() || p.components().any(|c| c.as_os_str().to_string_lossy().contains('/')) {
        p
    } else if !p.to_string_lossy().contains(std::path::MAIN_SEPARATOR) {
        // Bare filename → put in workspace.
        let workspace = std::env::var("DEFAULT_WORKSPACE")
            .unwrap_or_else(|_| "/home/ermer/.atomic_chat/".to_string());
        let mut wp = PathBuf::from(&workspace);
        wp.push(path);
        if let Err(e) = std::fs::create_dir_all(wp.parent().unwrap_or(Path::new("."))) {
            log::warn!("Failed to create directory for {}: {}", path, e);
        }
        wp
    } else {
        p
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// ReadFile
// ───────────────────────────────────────────────────────────────────────────────

pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str { "read_file" }
    fn description(&self) -> &str {
        "Read a file and return its contents with line numbers. Returns at most 200 lines per call."
    }

    async fn call(&self, _ctx: &ToolContext, params: &Value) -> ToolResult {
        let path_str = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("Missing required parameter: path"),
        };
        log::debug!("read_file: path={}", path_str);
        let path = resolve(&path_str);
        let start_line: usize = params.get("start_line").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(0);
        let end_line: Option<usize> = params.get("end_line").and_then(|v| v.as_u64()).map(|v| v as usize);

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("read_file: failed to read {}: {}", path.display(), e);
                return ToolResult::err(format!("{}: {}", e.kind(), e));
            }
        };
        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();
        let cap = start_line.saturating_add(READ_PAGE);
        let end = end_line.unwrap_or(cap).min(cap).min(total);
        let subset: Vec<&str> = lines[start_line..end].to_vec();
        let numbered: Vec<String> = subset.iter().enumerate().map(|(i, line)| {
            format!("{:>6}  {}", start_line + i + 1, line)
        }).collect();
        let has_more = end < total;

        let mut result = serde_json::json!({
            "path": path.display().to_string(),
            "content": numbered.join("\n"),
            "lines_returned": numbered.len(),
            "total_lines": total,
            "has_more": has_more,
        });
        if has_more {
            result["next_start_line"] = serde_json::json!(end);
        }
        ToolResult::ok_with_data(format!("Read {} lines from {}", numbered.len(), path.display()), result)
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// FileInfo
// ───────────────────────────────────────────────────────────────────────────────

pub struct InfoTool;

#[async_trait]
impl Tool for InfoTool {
    fn name(&self) -> &str { "file_info" }
    fn description(&self) -> &str { "Return metadata about a file: size, modified time, type, line count." }

    async fn call(&self, _ctx: &ToolContext, params: &Value) -> ToolResult {
        let path_str = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("Missing required parameter: path"),
        };
        log::debug!("file_info: path={}", path_str);
        let path = resolve(&path_str);
        let meta = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("file_info: failed to get metadata for {}: {}", path.display(), e);
                return ToolResult::err(format!("{}: {}", e.kind(), e));
            }
        };
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            log::warn!("file_info: could not read content of {}: {}", path.display(), e);
            String::new()
        });
        let line_count = content.lines().count();
        let ftype = if meta.is_dir() { "directory" } else { "file" };

        let modified = meta.modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
            .map(|d| chrono::DateTime::UNIX_EPOCH + chrono::TimeDelta::seconds(d.as_secs() as i64))
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_else(|| "unknown".to_string());

        ToolResult::ok_with_data("ok", serde_json::json!({
            "path": path.display().to_string(),
            "size": meta.len(),
            "modified_time": modified,
            "type": ftype,
            "line_count": line_count,
        }))
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// SummarizeFileContent
// ───────────────────────────────────────────────────────────────────────────────

pub struct SummaryTool;

#[async_trait]
impl Tool for SummaryTool {
    fn name(&self) -> &str { "summarize_file" }
    fn description(&self) -> &str { "Provide a structural summary of a file with line numbers for definitions." }

    async fn call(&self, _ctx: &ToolContext, params: &Value) -> ToolResult {
        let path_str = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("Missing required parameter: path"),
        };
        log::debug!("summarize_file: path={}", path_str);
        let path = resolve(&path_str);
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("summarize_file: failed to read {}: {}", path.display(), e);
                return ToolResult::err(format!("{}: {}", e.kind(), e));
            }
        };
        let lines: Vec<&str> = content.lines().collect();

        let import_re = Regex::new(r"^(?:use\s+|mod\s+|include!?)").unwrap();
        let class_re = Regex::new(r"^(\w*(?:impl|trait|mod|struct|enum|union))\s+(\w+)").unwrap();
        let func_re = Regex::new(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)").unwrap();

        let mut structure = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let trimmed_line = line.trim();
            if import_re.is_match(&trimmed_line) {
                structure.push(serde_json::json!({"type":"import","line":i+1,"content":&trimmed_line}));
            } else if let Some(caps) = class_re.captures(&trimmed_line) {
                structure.push(serde_json::json!({
                    "type": "definition",
                    "kind": &caps[1],
                    "name": &caps[2],
                    "line": i + 1,
                }));
            } else if let Some(caps) = func_re.captures(&trimmed_line) {
                structure.push(serde_json::json!({
                    "type": "function",
                    "name": &caps[1],
                    "line": i + 1,
                }));
            }
        }

        let classes = structure.iter().filter(|s| s["kind"].as_str().map_or(false, |k| k == "struct" || k == "enum" || k == "trait" || k == "union")).count();
        let functions = structure.iter().filter(|s| s["type"] == "function").count();
        let imports = structure.iter().filter(|s| s["type"] == "import").count();

        ToolResult::ok_with_data("ok", serde_json::json!({
            "path": path.display().to_string(),
            "line_count": lines.len(),
            "structure": structure,
            "summary": format!("File contains {} definitions, {} functions, {} imports.", classes, functions, imports),
        }))
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// WriteFile
// ───────────────────────────────────────────────────────────────────────────────

pub struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str { "write_file" }
    fn description(&self) -> &str { "Write content to a file. Defaults to append mode." }

    async fn call(&self, _ctx: &ToolContext, params: &Value) -> ToolResult {
        let path_str = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("Missing required parameter: path"),
        };
        let content = match params.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return ToolResult::err("Missing required parameter: content"),
        };
        let mode: String = params.get("mode").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| "append".to_string());
        log::debug!("write_file: path={}, mode={}", path_str, mode);
        let path = resolve(&path_str);

        // Create parent dirs.
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    log::warn!("write_file: failed to create parent dirs for {}: {}", path.display(), e);
                }
            }
        }

        let options = if mode == "overwrite" {
        std::fs::OpenOptions::new().write(true).truncate(true).open(&path)
        } else {
        std::fs::OpenOptions::new().write(true).append(true).open(&path)
        };

        let mut file = match options {
            Ok(f) => f,
            Err(e) => return ToolResult::err(format!("{}: {}", e.kind(), e)),
        };

        use std::io::Write;
        if let Err(e) = file.write_all(content.as_bytes()) {
            log::warn!("write_file: write error for {}: {}", path.display(), e);
            return ToolResult::err(format!("Write error: {}", e));
        }

        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or_else(|e| {
            log::warn!("write_file: could not read metadata for {}: {}", path.display(), e);
            0
        });
        log::debug!("write_file: wrote {} bytes to {} ({})", content.len(), path.display(), mode);
        ToolResult::ok_with_data("ok", serde_json::json!({
            "path": path.display().to_string(),
            "mode": mode,
            "total_size": size,
        }))
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// SearchFileForPattern (grep)
// ───────────────────────────────────────────────────────────────────────────────

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str { "grep" }
    fn description(&self) -> &str { "Search for patterns in files using ripgrep." }

    async fn call(&self, _ctx: &ToolContext, params: &Value) -> ToolResult {
        let path_str = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("Missing required parameter: path"),
        };
        let pattern = match params.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("Missing required parameter: pattern"),
        };
        log::debug!("grep: path={}, pattern={}", path_str, pattern);
        let path = resolve(&path_str);
        let case_sensitive: bool = params.get("case_sensitive").and_then(|v| v.as_bool()).unwrap_or(false);
        let max_results: usize = params.get("max_results").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(50);
        let glob = params.get("glob").and_then(|v| v.as_str()).map(|s| s.to_string());
        let max_results_str = max_results.to_string();

        let path_arg = match path.to_str() {
            Some(s) => s,
            None => return ToolResult::err(format!("Path contains non-UTF8 characters: {}", path.display())),
        };
        let mut cmd = vec!["rg", &pattern, path_arg, "-n", "-C2", "--max-count", &max_results_str];
        if !case_sensitive {
            cmd.push("-i");
        }
        if let Some(g) = &glob {
            if path.is_dir() {
                cmd.push("--glob");
                cmd.push(g);
            }
        }

        let output = match Command::new("rg").args(&cmd[1..]).output() {
            Ok(o) => o,
            Err(e) => {
                log::warn!("grep: ripgrep not found: {}", e);
                return ToolResult::err(format!("ripgrep not found: {}", e));
            }
        };

        let results_text = String::from_utf8_lossy(&output.stdout).to_string();
        let max_lines = max_results * 6;
        let lines: Vec<&str> = results_text.lines().collect();
        let truncated = lines.len() > max_lines;
        let output_text = lines[..lines.len().min(max_lines)].join("\n");

        ToolResult::ok_with_data("ok", serde_json::json!({
            "path": path.display().to_string(),
            "pattern": pattern,
            "case_sensitive": case_sensitive,
            "results": if output_text.is_empty() { "no matches found" } else { &output_text },
            "truncated": truncated,
        }))
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// FindFnDefInFile
// ───────────────────────────────────────────────────────────────────────────────

pub struct FindDefinitionTool;

#[async_trait]
impl Tool for FindDefinitionTool {
    fn name(&self) -> &str { "find_definition" }
    fn description(&self) -> &str { "Find function and class definitions in a file." }

    async fn call(&self, _ctx: &ToolContext, params: &Value) -> ToolResult {
        let path_str = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("Missing required parameter: path"),
        };
        log::debug!("find_definition: path={}", path_str);
        let path = resolve(&path_str);
        let name_filter = params.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());
        let def_type: String = params.get("def_type").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| "any".to_string());

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("find_definition: failed to read {}: {}", path.display(), e);
                return ToolResult::err(format!("{}: {}", e.kind(), e));
            }
        };
        let lines: Vec<&str> = content.lines().collect();

        let func_re = Regex::new(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)").unwrap();
        let class_re = Regex::new(r"^(?:pub\s+)?(struct|enum|trait|union|impl)\s+(\w+)").unwrap();

        let mut results = Vec::new();

        let search_funcs = def_type == "any" || def_type == "function";
        let search_classes = def_type == "any" || def_type == "class";

        if search_funcs {
            for (i, line) in lines.iter().enumerate() {
                if let Some(caps) = func_re.captures(line.trim()) {
                    let fname = &caps[1];
                    if let Some(ref nf) = name_filter {
                        if fname != nf.as_str() { continue; }
                    }
                    let end = self._find_def_end(&lines, i);
                    results.push(serde_json::json!({
                        "type": "function",
                        "name": fname,
                        "line_start": i + 1,
                        "line_end": end,
                        "content_preview": line.trim(),
                    }));
                }
            }
        }

        if search_classes {
            for (i, line) in lines.iter().enumerate() {
                if let Some(caps) = class_re.captures(line.trim()) {
                    let kind = &caps[1];
                    let cname = caps.get(2).map(|m| m.as_str()).unwrap_or("");
                    if let Some(ref nf) = name_filter {
                        if cname != nf.as_str() { continue; }
                    }
                    let end = self._find_def_end(&lines, i);
                    results.push(serde_json::json!({
                        "type": kind,
                        "name": cname,
                        "line_start": i + 1,
                        "line_end": end,
                        "content_preview": line.trim(),
                    }));
                }
            }
        }

        if results.is_empty() {
            let target = name_filter.as_deref().map(|n| format!("\"{}\"", n)).unwrap_or_else(|| "any definitions".to_string());
            return ToolResult::err(format!("No definition found for {} in {}", target, path.display()));
        }

        ToolResult::ok_with_data("ok", serde_json::json!({
            "path": path.display().to_string(),
            "definitions_found": results.len(),
            "results": results,
        }))
    }
}

impl FindDefinitionTool {
    fn _find_def_end(&self, lines: &[&str], start_idx: usize) -> usize {
        let base_indent = lines[start_idx].chars().take_while(|source_character| source_character.is_whitespace()).count();
        let def_re = Regex::new(r"^(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|union|impl)\s+\w+").unwrap();
        let mut end_idx = start_idx;
        for i in (start_idx + 1)..lines.len() {
            let trimmed_source_line = lines[i].trim();
            if trimmed_source_line.is_empty() { continue; }
            let current_indent = lines[i].chars().take_while(|character| character.is_whitespace()).count();
            if current_indent <= base_indent && !trimmed_source_line.is_empty() {
                if def_re.is_match(&trimmed_source_line) || current_indent <= base_indent {
                    end_idx = i;
                    break;
                }
            }
            end_idx = i;
        }
        end_idx + 1
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// ListFilesInDirectory
// ───────────────────────────────────────────────────────────────────────────────

pub struct ListDirTool;

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str { "list_dir" }
    fn description(&self) -> &str { "List files and directories in the provided directory." }

    async fn call(&self, _ctx: &ToolContext, params: &Value) -> ToolResult {
        let path_str = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("Missing required parameter: path"),
        };
        log::debug!("list_dir: path={}", path_str);
        let path = resolve(&path_str);
        let entries: Vec<String> = match std::fs::read_dir(&path) {
            Ok(rd) => rd.filter_map(|e| e.ok().map(|entry| entry.file_name().to_string_lossy().to_string())).collect(),
            Err(e) => {
                log::warn!("list_dir: failed to read {}: {}", path.display(), e);
                return ToolResult::err(format!("{}: {}", e.kind(), e));
            }
        };
        log::debug!("list_dir: found {} entries in {}", entries.len(), path.display());
        ToolResult::ok_with_data("ok", serde_json::json!({ "entries": entries }))
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// MoveFileOrDirectory
// ───────────────────────────────────────────────────────────────────────────────

pub struct MoveTool;

#[async_trait]
impl Tool for MoveTool {
    fn name(&self) -> &str { "move_file" }
    fn description(&self) -> &str { "Move or rename a file or directory." }

    async fn call(&self, _ctx: &ToolContext, params: &Value) -> ToolResult {
        let src_str = match params.get("source_path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("Missing required parameter: source_path"),
        };
        let dst_str = match params.get("dest_path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("Missing required parameter: dest_path"),
        };
        log::debug!("move_file: src={}, dst={}", src_str, dst_str);
        let src = resolve(&src_str);
        let dst = resolve(&dst_str);

        if !src.exists() {
            log::warn!("move_file: source not found: {}", src.display());
            return ToolResult::err(format!("Source not found: {}", src.display()));
        }

        // Check extension change guard.
        if !src.is_dir() {
            let src_ext = src.extension().map(|e| e.to_string_lossy().to_string());
            let dst_ext = dst.extension().map(|e| e.to_string_lossy().to_string());
            if src_ext != dst_ext && src_ext.as_ref().map(|e| !e.is_empty()).unwrap_or(false) {
                return ToolResult::err("Changing file extension only allowed on previously empty extensions.".to_string());
            }
        }

        if let Some(parent) = dst.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    log::warn!("move_file: failed to create parent dirs for {}: {}", dst.display(), e);
                }
            }
        }

        match std::fs::rename(&src, &dst) {
            Ok(()) => {
                log::debug!("move_file: moved {} → {}", src.display(), dst.display());
                ToolResult::ok_with_data("ok", serde_json::json!({
                    "source": src.display().to_string(),
                    "dest": dst.display().to_string(),
                    "status": "success",
                }))
            }
            Err(e) => {
                log::warn!("move_file: rename failed {} → {}: {}", src.display(), dst.display(), e);
                ToolResult::err(format!("{}: {}", e.kind(), e))
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// FindFileDirByName (fdfind)
// ───────────────────────────────────────────────────────────────────────────────

pub struct FindTool;

#[async_trait]
impl Tool for FindTool {
    fn name(&self) -> &str { "find" }
    fn description(&self) -> &str { "Find files/dirs using a query." }

    async fn call(&self, _ctx: &ToolContext, params: &Value) -> ToolResult {
        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) => q.to_string(),
            None => return ToolResult::err("Missing required parameter: query"),
        };
        log::debug!("find: query={}", query);
        let path = params.get("path").and_then(|v| v.as_str()).map(|s| resolve(s));
        let target_type = params.get("target_type").and_then(|v| v.as_str()).map(|s| s.to_string());
        let extensions: Vec<&str> = params.get("extension")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let mut cmd = vec!["fdfind", &query];
        if let Some(ref p) = path {
            match p.to_str() {
                Some(s) => cmd.push(s),
                None => return ToolResult::err(format!("Path contains non-UTF8 characters: {}", p.display())),
            }
        }
        if let Some(ref tt) = target_type {
            match tt.as_str() {
                "file" => cmd.extend(["-t", "f"]),
                "directory" => cmd.extend(["-t", "d"]),
                _ => return ToolResult::err("Invalid target_type. Must be one of: [file,directory].".to_string()),
            }
        }
        for ext in &extensions {
            cmd.push("-e");
            cmd.push(ext);
        }

        let output = match Command::new("fdfind").args(&cmd[1..]).output() {
            Ok(o) => o,
            Err(e) => {
                log::warn!("find: fdfind not found: {}", e);
                return ToolResult::err(format!("fdfind not found: {}", e));
            }
        };

        let data = String::from_utf8_lossy(&output.stdout).to_string();
        if data.trim().is_empty() {
            log::debug!("find: no results for query '{}'", query);
            return ToolResult::ok("No results.");
        }
        log::debug!("find: {} result(s) for query '{}'", data.lines().count(), query);
        ToolResult::ok(data)
    }
}

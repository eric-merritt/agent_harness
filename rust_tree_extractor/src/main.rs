use serde::Serialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum FileItem {
	Function { name: String },
	Struct { name: String },
	Enum { name: String },
	Trait { name: String },
}

#[derive(Serialize, Debug)]
#[serde(tag = "node_type", rename_all = "snake_case")]
enum Node {
	Directory { name: String, children: Vec<Node> },
	File { name: String, items: Vec<FileItem> },
}

fn parse_rust_file(path: &Path) -> Vec<FileItem> {
	let mut file_items = Vec::new();

	let content = match fs::read_to_string(path) {
		Ok(c) => c,
		Err(_) => return file_items,
	};

	let syntax_tree = match syn::parse_file(&content) {
		Ok(ast) => ast,
		Err(_) => return file_items,
	};

	for item in syntax_tree.items {
		match item {
			syn::Item::Fn(func) => {
				file_items.push(FileItem::Function {
					name: func.sig.ident.to_string(),
				});
			}
			syn::Item::Struct(strct) => {
				file_items.push(FileItem::Struct {
					name: strct.ident.to_string(),
				});
			}
			syn::Item::Enum(enm) => {
				file_items.push(FileItem::Enum {
					name: enm.ident.to_string(),
				});
			}
			syn::Item::Trait(trt) => {
				file_items.push(FileItem::Trait {
					name: trt.ident.to_string(),
				});
			}
			_ => {}
		}
	}

	file_items
}

fn build_tree(path: &Path) -> Option<Node> {
	let file_name = path
		.file_name()
		.map(|n| n.to_string_lossy().into_owned())
		.unwrap_or_else(|| "/".to_string());

	if path.is_dir() {
		if file_name == "target"
			|| file_name == ".git"
			|| file_name == ".cache"
			|| file_name == "rust_tree_extractor"
		{
			return None;
		}

		let mut children = Vec::new();
		if let Ok(entries) = fs::read_dir(path) {
			for entry in entries.flatten() {
				if let Some(child_node) = build_tree(&entry.path()) {
					children.push(child_node);
				}
			}
		}

		if children.is_empty() {
			return None;
		}

		Some(Node::Directory {
			name: file_name,
			children,
		})
	} else if path.extension().map_or(false, |ext| ext == "rs") {
		let items = parse_rust_file(path);
		Some(Node::File {
			name: file_name,
			items,
		})
	} else {
		None
	}
}

fn main() {
	let args: Vec<String> = env::args().collect();

	// Capture the exact location your terminal is currently sitting in
	let current_dir = env::current_dir().expect("Failed to detect current working directory");

	let path_str = if args.len() > 1 { &args[1] } else { "." };

	let target_path = Path::new(path_str);
	let absolute_path = fs::canonicalize(target_path).unwrap_or_else(|_| target_path.to_path_buf());

	println!(
		"Analyzing codebase structure tree path: {:?}",
		absolute_path
	);

	if let Some(root_node) = build_tree(&absolute_path) {
		let json_output = serde_json::to_string_pretty(&root_node)
			.expect("Failed to serialize AST tree block mapping");

		// CRITICAL FIX: Explicitly target your terminal CWD instead of the package directory
		let output_file_path = current_dir.join("codebase_structure.json");

		fs::write(&output_file_path, json_output)
			.expect("Unable to write JSON output mapping data to disk");

		println!(
			"Successfully generated structure map at: {:?}",
			output_file_path
		);
	} else {
		eprintln!(
			"Error: Could not evaluate any valid .rs source files inside target path: {:?}",
			absolute_path
		);
	}
}

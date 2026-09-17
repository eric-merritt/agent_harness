import json
import re
import sys


def extract_rust_pairs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pairs = []
    
    # State tracking
    passed_initial_setup = False
    current_comment_lines = []
    in_block_comment = False

    # Regex patterns
    initial_keywords = re.compile(r"^\s*(use|mod|extern|pub\s+mod)\b")
    fn_start_pattern = re.compile(r"^\s*(pub\s+)?(async\s+)?fn\b")
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 1. Track if we have passed the initial file imports/modules
        if not passed_initial_setup:
            if initial_keywords.match(line) or fn_start_pattern.match(line):
                passed_initial_setup = True

        # 2. Parse Multi-line Block Comments (/* ... */)
        if in_block_comment:
            current_comment_lines.append(line)
            if "*/" in stripped:
                in_block_comment = False
            i += 1
            continue
        
        if passed_initial_setup and stripped.startswith("/*"):
            current_comment_lines = [line]
            if "*/" not in stripped:
                in_block_comment = True
            i += 1
            continue

        # 3. Parse Single-line Comments (// or /// docstrings)
        if passed_initial_setup and stripped.startswith("//"):
            current_comment_lines.append(line)
            i += 1
            continue

        # 4. Check if the comment block is immediately followed by a function
        if current_comment_lines:
            # If the next non-empty line starts a function
            if stripped and fn_start_pattern.match(line):
                fn_lines = []
                brace_count = 0
                started_body = False
                
                # Consume the full function block by tracking brackets
                while i < len(lines):
                    fn_line = lines[i]
                    fn_lines.append(fn_line)
                    
                    brace_count += fn_line.count("{")
                    brace_count -= fn_line.count("}")
                    
                    if "{" in fn_line:
                        started_body = True
                        
                    # Stop once the main function scope brackets close safely
                    if started_body and brace_count <= 0:
                        break
                    i += 1

                # Clean up and pair the texts together
                comment_text = "".join(current_comment_lines).strip()
                function_text = "".join(fn_lines).strip()
                
                pairs.append({
                    "comment": comment_text,
                    "function": function_text
                })
            
            # Reset comment accumulator if it wasn't followed by a function
            if stripped or line == "\n":
                current_comment_lines = []

        i += 1

    return pairs


def save_to_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gen_cal.py <input_file.rs> <output_file.jsonl>")
        sys.exit(1)

    input_rs = sys.argv[1]
    output_jsonl = sys.argv[2]

    try:
        extracted_data = extract_rust_pairs(input_rs)
        save_to_jsonl(extracted_data, output_jsonl)
        print(f"Successfully extracted {len(extracted_data)} Rust pairs into '{output_jsonl}'.")
    except Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)

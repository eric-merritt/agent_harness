import json
import re
import sys
import traceback


def extract_moderncv_groups(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    structured_blocks = []
    
    current_section = "Preamble"
    current_subsection = ""
    
    # Matches \section{...} or \subsection{...}
    section_pattern = re.compile(r"^\s*\\(section|subsection)\s*\{([^}]+)\}")
    
    # Matches standard moderncv skill/item entry macros
    macro_pattern = re.compile(
        r"^\s*\\(cvitem|cvcomputer|cvdoubleitem|cvlistitem|cvlistdoubleitem|cvline|cventry)\b"
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip completely empty lines or line-level comments
        if not stripped or stripped.startswith("%"):
            i += 1
            continue

        # 1. Track Sections and Subsections
        sec_match = section_pattern.match(stripped)
        if sec_match:
            level = sec_match.group(1)
            name = sec_match.group(2).strip()
            if level == "section":
                current_section = name
                current_subsection = ""
            else:
                current_subsection = name
            i += 1
            continue

        # 2. Extract Skill & Item Macro Groups
        if macro_pattern.match(stripped):
            macro_lines = []
            brace_count = 0
            started_braces = False
            
            # Consume multi-line curly brace structures safely
            while i < len(lines):
                fn_line = lines[i]
                macro_lines.append(fn_line)
                
                # Strip out inline comments to prevent bad brace counts
                clean_fn_line = fn_line.split("%")[0]
                
                brace_count += clean_fn_line.count("{")
                brace_count -= clean_fn_line.count("}")
                
                if "{" in clean_fn_line:
                    started_braces = True
                    
                if started_braces and brace_count <= 0:
                    break
                i += 1

            macro_text = "\n".join(macro_lines).strip()
            
            # Append item tied cleanly to its structural category
            structured_blocks.append({
                "section": current_section,
                "subsection": current_subsection,
                "entry": macro_text
            })
            
            i += 1
            continue

        i += 1

    return structured_blocks


def save_to_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_moderncv.py <input_file.tex> <output_file.jsonl>")
        sys.exit(1)

    input_tex = sys.argv[1]
    output_jsonl = sys.argv[2]

    try:
        extracted_data = extract_moderncv_groups(input_tex)
        save_to_jsonl(extracted_data, output_jsonl)
        print(f"Successfully extracted {len(extracted_data)} ModernCV elements into '{output_jsonl}'.")
    except Exception as e:
        print(f"Error parsing LaTeX file: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

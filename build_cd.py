import json
import sys
import torch
from transformers import AutoTokenizer
from safetensors.torch import save_file

def build_calibration_dataset(jsonl_paths, output_safetensors, model_id, max_seq_len=4096):
    # 1. Load the target model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids_list = []

    # 2. Extract sequences from all listed files
    for path in jsonl_paths:
        print(f"Tokenizing input from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    
                    # FIX: Use .get() with fallback empty strings to handle missing keys safely
                    sec = data.get("section", "")
                    sub = data.get("subsection", "")
                    entry = data.get("entry", data.get("cv_entry", data.get("comment", "")))
                    
                    # Skip rows that are completely empty of meaningful text fields
                    if not sec and not sub and not entry:
                        continue
                        
                    # Combine sections into raw text blocks
                    text_payload = f"Sec: {sec} | Sub: {sub} | Entry: {entry}"
                    
                    # Encode text into plain tokens
                    encoded = tokenizer(
                        text_payload,
                        truncation=True,
                        max_length=max_seq_len,
                        return_tensors="pt"
                    )
                    
                    input_ids_list.append(encoded["input_ids"])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line {idx} in {path}")
                    continue

    if not input_ids_list:
        print("Error: No data successfully tokenized.")
        return

    # 3. Pad all sequences out to the block maximum length so they match shapes
    padded_input_ids = []
    for ids in input_ids_list:
        seq_len = ids.shape[1]
        if seq_len < max_seq_len:
            padding = torch.full((1, max_seq_len - seq_len), tokenizer.pad_token_id, dtype=torch.long)
            ids = torch.cat([ids, padding], dim=1)
        padded_input_ids.append(ids)

    # Compile the final binary payload
    tensors = {
        "input_ids": torch.cat(padded_input_ids, dim=0)
    }

    # 4. Stream to zero-copy safetensors format
    save_file(tensors, output_safetensors)
    print(f"Success! Built calibration dataset at '{output_safetensors}' with shape {tensors['input_ids'].shape}.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python build_cd.py <output.safetensors> <model_id_or_path> <file1.jsonl> [file2.jsonl ...]")
        sys.exit(1)
        
    out_file = sys.argv[1]
    model_path = sys.argv[2]
    input_files = sys.argv[3:]
    
    build_calibration_dataset(input_files, out_file, model_path)

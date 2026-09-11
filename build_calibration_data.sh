#!/bin/bash

# Define output location
OUTPUT_FILE="$HOME/qwen_custom_calibration.txt"
clear_existing=true

if [ "$clear_existing" = true ] ; then
    echo "[-] Clearing old calibration file..."
    rm -f "$OUTPUT_FILE"
fi

echo "[+] Harvesting rust memory_controller module files..."
# Find all .rs files inside your memory_controller module and append them
if [ -d "src/memory_controller" ]; then
    find src/memory_controller -name "*.rs" -type f | while read -r file; do
        echo -e "\n\n// --- FILE: $file ---\n" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
    done
else
    echo "[!] Warning: src/memory_controller directory not found!"
fi

echo "[+] Harvesting GLSL compute shader..."
# Append your new hessian_estimate compute shader
if [ -f "src/models/hessian_estimate.comp" ]; then
    echo -e "\n\n// --- FILE: src/models/hessian_estimate.comp ---\n" >> "$OUTPUT_FILE"
    cat "src/models/hessian_estimate.comp" >> "$OUTPUT_FILE"
elif [ -f "hessian_estimate.comp" ]; then
    echo -e "\n\n// --- FILE: hessian_estimate.comp ---\n" >> "$OUTPUT_FILE"
    cat "hessian_estimate.comp" >> "$OUTPUT_FILE"
else
    echo "[!] Warning: hessian_estimate.comp not found!"
fi

# Statistical check for your dataset size
if [ -f "$OUTPUT_FILE" ]; then
    line_count=$(wc -l < "$OUTPUT_FILE")
    char_count=$(wc -m < "$OUTPUT_FILE")
    # Rough token estimation for code: ~3.5 characters per token
    approx_tokens=$((char_count / 3))
    
    echo "----------------------------------------"
    echo "[+] Custom Dataset Created Successfully!"
    echo "Location: $OUTPUT_FILE"
    echo "Total Lines: $line_count"
    echo "Approximate Tokens: $approx_tokens"
    echo "----------------------------------------"
else
    echo "[X] Error: Failed to generate calibration file."
fi

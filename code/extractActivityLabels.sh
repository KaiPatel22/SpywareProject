#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <text_file>"
    exit 1
fi

TEXT_FILE="$1"
OUTPUT_FILE="${TEXT_FILE%.txt}_activity.py"

echo "[*] Processing $TEXT_FILE"
echo "[*] Output: $OUTPUT_FILE"

{
    echo "activities = ["
    
    grep -E '^[0-9]+(\.[0-9]+)? to [0-9]+(\.[0-9]+)? -' "$TEXT_FILE" | while read line; do
        start=$(echo "$line" | awk '{print $1}')
        end=$(echo "$line" | awk '{print $3}')
        label=$(echo "$line" | sed 's/^.* - //')
        
        echo -e "\t{\"start\": $start, \"end\": $end, \"label\": \"$label\"},"
    done
    
    echo "]"
} > "$OUTPUT_FILE"

echo "[+] Done. Created $OUTPUT_FILE"
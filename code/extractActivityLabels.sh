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
    
    grep -E '^[0-9]+\.[0-9]+ to [0-9]+\.[0-9]+ -' "$TEXT_FILE" | while read line; do
        start=$(echo "$line" | awk '{print $1}')
        end=$(echo "$line" | awk '{print $3}')
        label=$(echo "$line" | sed 's/^[0-9.]*  *to  *[0-9.]*  *-  *//' | awk '{for(i=1;i<=NF;i++){if(i==1) printf tolower($i); else printf toupper(substr($i,1,1)) tolower(substr($i,2))} print ""}')
        
        echo -e "\t{\"start\": $start, \"end\": $end, \"label\": \"$label\"},"
    done
    
    echo "]"
} > "$OUTPUT_FILE"

echo "[+] Done. Created $OUTPUT_FILE"
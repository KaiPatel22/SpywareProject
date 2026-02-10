#!/bin/bash

# -----------------------------
# Usage check
# -----------------------------
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <pcap_file>"
    exit 1
fi

PCAP_FILE="$1"

# Output filename
OUTPUT_FILE="${PCAP_FILE%.pcap}_extracted.csv"

echo "[*] Processing $PCAP_FILE"
echo "[*] Output: $OUTPUT_FILE"

# -----------------------------
# tshark extraction
# -----------------------------
tshark -r "$PCAP_FILE" \
-T fields \
-e frame.time_epoch \
-e ip.src \
-e ip.dst \
-e ip.ttl \
-e tcp.flags \
-e tcp.window_size_value \
-e ip.proto \
-e frame.len \
-e tcp.srcport \
-e tcp.dstport \
-e udp.srcport \
-e udp.dstport \
-E header=y \
-E separator=, \
> "$OUTPUT_FILE"

echo "[+] Done."


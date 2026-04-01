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
-Y "(ip.addr == 192.168.0.52 || ip.addr == 192.168.0.200 || ip.addr == 192.168.0.47 || ip.addr == 192.168.0.55 || ip.addr == 192.168.0.43 || ip.addr == 192.168.0.35 || ip.addr == 192.168.0.59) && !(ip.addr == 192.168.0.69) && !(ip.addr == 192.168.0.187) && !(ip.addr == 192.168.0.54) && !tcp.analysis.retransmission && !tcp.analysis.duplicate_ack" \
-e frame.time_epoch \
-e frame.time_delta \
-e frame.len \
-e ip.src \
-e ip.dst \
-e ip.proto \
-e ip.ttl \
-e ip.len \
-e tcp.len \
-e tcp.flags \
-e tcp.time_relative \
-e tcp.stream \
-e tcp.analysis.ack_rtt \
-e tcp.time_delta \
-e tcp.srcport \
-e tcp.dstport \
-e udp.srcport \
-e udp.dstport \
-e dns.qry.name \
-e tcp.window_size_value \
-e tls.record.length \
-e tls.handshake.type \
-e tls.record.content_type \
-e udp.length \
-E header=y \
-E separator=, \
> "$OUTPUT_FILE"

echo "[+] Done."


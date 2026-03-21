import sys
import random
import pandas as pd

DEVICE_IPS = {
    'loungeBulb': '192.168.0.52',
    'bedroomBulb': '192.168.0.47',
    'camera': '192.168.0.55',
    'hub': '192.168.0.200',
    'plug': '192.168.0.59'
}

def loadActivities(filename : str) -> dict:
    listOfActivities = {}
    with open(filename, 'r') as f:
        exec(f.read(), listOfActivities)
    return listOfActivities['activities']

def loadDf(filename: str) -> pd.DataFrame:
    numericCols = ['frame.time_epoch', 'frame.time_delta', 'frame.len', 'ip.proto', 'ip.ttl', 'ip.len', 'tcp.len', 'tcp.stream', 'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport', 'tcp.window_size_value', 'tls.handshake.type', 'tls.record.length', 'udp.length']

    df = pd.read_csv(filename, escapechar="\\", engine="python", on_bad_lines="warn").sort_values('frame.time_epoch').reset_index(drop=True)

    for col in numericCols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['tcp.flags'] = df['tcp.flags'].dropna().apply(lambda x: int(str(x), 16) if str(x).startswith('0x') else int(x)).astype('Int64')

    df["ip.src"] = df["ip.src"].fillna("").astype(str).str.strip()
    df["ip.dst"] = df["ip.dst"].fillna("").astype(str).str.strip()
    df["dns.qry.name"] = df["dns.qry.name"].fillna("").astype(str).str.strip()

    return df

def extractFeatures(window: pd.DataFrame, windowID: int, windowStart: float, windowEnd: float) -> dict:
    """Extract statistical features from a slice of the packet dataframe."""
    WINDOW_SIZE = windowEnd - windowStart
    n = len(window)

    tcpPackets = window[window['ip.proto'] == 6]
    udpPackets = window[window['ip.proto'] == 17]
    tcpCount = len(tcpPackets)
    udpCount = len(udpPackets)

    flags = tcpPackets['tcp.flags'].dropna().astype('int64')
    synCount    = int((flags & 0x002).gt(0).sum())
    ackCount    = int((flags & 0x010).gt(0).sum())
    finCount    = int((flags & 0x001).gt(0).sum())
    rstCount    = int((flags & 0x004).gt(0).sum())
    pshCount    = int((flags & 0x008).gt(0).sum())
    synOnlyCount = int(((flags & 0x002).gt(0) & (flags & 0x010).eq(0)).sum())

    frameLengths = window['frame.len']
    interArrivalTimes = window['frame.time_delta']
    dns_non_empty = window["dns.qry.name"][window["dns.qry.name"] != ""]

    devicePacketCounts = {}
    for device, ip in DEVICE_IPS.items():
        devicePacketCounts[f'packetTo_{device}']   = int((window['ip.dst'] == ip).sum())
        devicePacketCounts[f'packetFrom_{device}'] = int((window['ip.src'] == ip).sum())

    return {
        'windowID': windowID,
        'windowStart': windowStart,
        'windowEnd': windowEnd,

        'packetCount': n,
        'tcpPacketCount': tcpCount,
        'udpPacketCount': udpCount,
        'tcpRatio': tcpCount / n,
        'udpRatio': udpCount / n,

        'avgPacketLength': frameLengths.mean(),
        'stdPacketLength': frameLengths.std(),
        'minPacketLength': frameLengths.min(),
        'maxPacketLength': frameLengths.max(),
        'medianPacketLength': frameLengths.median(),
        'smallPacketCount': int((frameLengths < 100).sum()),
        'largePacketCount': int((frameLengths > 500).sum()),
        'throughput': frameLengths.sum() / WINDOW_SIZE,

        'uniqueSrcIPs': window['ip.src'].nunique(),
        'uniqueDstIPs': window['ip.dst'].nunique(),
        'avgTTL': window['ip.ttl'].mean(),
        'stdTTL': window['ip.ttl'].std(),
        'avgIPLen': window['ip.len'].mean(),
        'stdIPLen': window['ip.len'].std(),

        'uniqueTCPSrcPorts': window['tcp.srcport'].dropna().nunique(),
        'uniqueTCPDstPorts': window['tcp.dstport'].dropna().nunique(),
        'avgTCPLen': tcpPackets['tcp.len'].mean(),
        'stdTCPLen': tcpPackets['tcp.len'].std(),
        'tcpPayloadPacketCount': int((tcpPackets['tcp.len'] > 0).sum()),
        'tcpPayloadPacketRatio': (tcpPackets['tcp.len'] > 0).sum() / tcpCount if tcpCount > 0 else 0,
        'uniqueTCPStreams': window['tcp.stream'].dropna().nunique(),
        'avgTCPWindowSize': window['tcp.window_size_value'].dropna().mean(),
        'minTCPWindowSize': window['tcp.window_size_value'].dropna().min(),

        'tcpSynCount': synCount,
        'tcpAckCount': ackCount,
        'tcpFinCount': finCount,
        'tcpRstCount': rstCount,
        'tcpPshCount': pshCount,
        'tcpSynOnlyCount': synOnlyCount,

        'uniqueUDPSrcPorts': window['udp.srcport'].dropna().nunique(),
        'uniqueUDPDstPorts': window['udp.dstport'].dropna().nunique(),

        'avgInterArrivalTime': interArrivalTimes.mean(),
        'stdInterArrivalTime': interArrivalTimes.std(),
        'minInterArrivalTime': interArrivalTimes.min(),
        'maxInterArrivalTime': interArrivalTimes.max(),

        **devicePacketCounts,

        'tlsHandshakeCount': int((window['tls.handshake.type'].dropna() == 1).sum()),
        'avgTLSRecordLen': window['tls.record.length'].dropna().mean(),
        'avgUDPLen': window['udp.length'].dropna().mean(),

        'DNSQueryCount': len(dns_non_empty),
        'uniqueDNSQueries': dns_non_empty.nunique()
    }

# ── Sliding window (original approach) ────────────────────────────────────────

def createWindows(df: pd.DataFrame) -> pd.DataFrame:
    WINDOW_SIZE = 5
    STEP = WINDOW_SIZE

    startTime = df['frame.time_epoch'].min()
    endTime   = df['frame.time_epoch'].max()

    windows = []
    windowID = 0
    time = startTime

    while time + WINDOW_SIZE <= endTime:
        windowStart = time
        windowEnd   = time + WINDOW_SIZE
        window = df[(df['frame.time_epoch'] >= windowStart) & (df['frame.time_epoch'] < windowEnd)]

        if len(window) > 0:
            windows.append(extractFeatures(window, windowID, windowStart, windowEnd))

        time += STEP
        windowID += 1

    return pd.DataFrame(windows)

def labelWindows(df: pd.DataFrame, activities: list) -> pd.DataFrame:
    def getLabel(row):
        OVERLAP_THRESHOLD = 0.2
        start    = row["windowStart"]
        end      = row["windowEnd"]
        duration = end - start

        for activity in activities:
            overlapStart    = max(start, activity["start"])
            overlapEnd      = min(end, activity["end"])
            overlapDuration = max(0, overlapEnd - overlapStart)

            if overlapDuration / duration >= OVERLAP_THRESHOLD:
                return activity["label"]
        return "idle"

    df["label"] = df.apply(getLabel, axis=1)
    return df

# ── Event-centered windows (new approach) ─────────────────────────────────────

def createEventCenteredWindows(df: pd.DataFrame, activities: list, radius: float = 0.5, idlePerEvent: int = 3) -> pd.DataFrame:
    """
    For each activity, create a window centered on the event midpoint.
    Also samples `idlePerEvent` idle windows from quiet gaps between events.

    radius       : half-width of each window in seconds (default 0.5 → 1s window)
    idlePerEvent : how many idle windows to sample per activity (controls balance)
    """
    rows = []
    windowID = 0

    # Sort activities by start time so gap detection is straightforward
    sorted_acts = sorted(activities, key=lambda a: a["start"])

    # ── labeled event windows ──────────────────────────────────────────────────
    for activity in sorted_acts:
        center = (activity["start"] + activity["end"]) / 2
        windowStart = center - radius
        windowEnd   = center + radius

        window = df[(df['frame.time_epoch'] >= windowStart) & (df['frame.time_epoch'] < windowEnd)]

        if len(window) > 0:
            row = extractFeatures(window, windowID, windowStart, windowEnd)
            row["label"] = activity["label"]
            rows.append(row)
            windowID += 1

    # ── idle windows sampled from gaps ────────────────────────────────────────
    captureStart = df['frame.time_epoch'].min()
    captureEnd   = df['frame.time_epoch'].max()
    windowSize   = radius * 2

    # Build list of quiet intervals: gaps between activity events (plus before/after)
    quietIntervals = []
    boundaries = [captureStart] + [a["start"] for a in sorted_acts] + [a["end"] for a in sorted_acts] + [captureEnd]
    boundaries.sort()

    # Pair up gap start/end, skipping intervals that overlap any activity
    gapStart = captureStart
    for act in sorted_acts:
        gapEnd = act["start"] - radius  # leave a radius buffer before each event
        if gapEnd - gapStart >= windowSize:
            quietIntervals.append((gapStart, gapEnd))
        gapStart = act["end"] + radius  # leave a radius buffer after each event

    if captureEnd - gapStart >= windowSize:
        quietIntervals.append((gapStart, captureEnd))

    # Sample idle window centers uniformly from quiet intervals
    totalQuietDuration = sum(e - s for s, e in quietIntervals)
    idleTargetCount = len(activities) * idlePerEvent
    random.seed(42)

    idleSampled = 0
    attempts = 0
    while idleSampled < idleTargetCount and attempts < idleTargetCount * 20:
        attempts += 1
        # Pick a random point weighted by interval length
        pick = random.uniform(0, totalQuietDuration)
        cumulative = 0
        for ivStart, ivEnd in quietIntervals:
            ivLen = ivEnd - ivStart
            if pick <= cumulative + ivLen:
                center = ivStart + (pick - cumulative)
                windowStart = center - radius
                windowEnd   = center + radius
                window = df[(df['frame.time_epoch'] >= windowStart) & (df['frame.time_epoch'] < windowEnd)]
                if len(window) > 0:
                    row = extractFeatures(window, windowID, windowStart, windowEnd)
                    row["label"] = "idle"
                    rows.append(row)
                    windowID += 1
                    idleSampled += 1
                break
            cumulative += ivLen

    result = pd.DataFrame(rows)
    print(f"Event windows : {(result['label'] != 'idle').sum()}")
    print(f"Idle windows  : {(result['label'] == 'idle').sum()}")
    print(result["label"].value_counts())
    return result

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python windowAnalysis2.py <packet csv file> <python activity file> [--sliding]")
        print("  Default: event-centered windows")
        print("  --sliding: use original sliding window approach")
        sys.exit(1)

    csvFile      = sys.argv[1]
    activityFile = sys.argv[2]
    useSlidingWindow = "--sliding" in sys.argv

    activities = loadActivities(activityFile)
    df = loadDf(csvFile)

    if useSlidingWindow:
        windows = createWindows(df)
        labeledWindows = labelWindows(windows, activities)
    else:
        labeledWindows = createEventCenteredWindows(df, activities, radius=0.5, idlePerEvent=3)

    output = csvFile.replace('.csv', '_windows.csv')
    labeledWindows.to_csv(output, index=False)
    print(f"Saved to {output}")
    print(labeledWindows.head())

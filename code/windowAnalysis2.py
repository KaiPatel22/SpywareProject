import sys
import pandas as pd

def loadActivities(filename : str) -> dict:
    listOfActivities = {}
    with open(filename, 'r') as f:
        exec(f.read(), listOfActivities)
    return listOfActivities['activities']

def createWindows(filename : str) -> pd.DataFrame:
    WINDOW_SIZE = 5
    STEP = WINDOW_SIZE

    DEVICE_IPS = {
        'loungeBulb': '192.168.0.52',
        'bedroomBulb': '192.168.0.47',
        'camera': '192.168.0.55',
        'hub': '192.168.0.200',
        'plug': '192.168.0.59'
    }

    df = pd.read_csv(filename, escapechar="\\", engine="python", on_bad_lines="warn").sort_values('frame.time_epoch').reset_index(drop=True)

    numericCols = ['frame.time_epoch', 'frame.time_delta', 'frame.len', 'ip.proto', 'ip.ttl', 'ip.len', 'tcp.len', 'tcp.stream' ,'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport', 'tcp.window_size_value', 'tls.handshake.type', 'tls.record.length', 'udp.length']

    for col in numericCols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['tcp.flags'] = df['tcp.flags'].dropna().apply(lambda x: int(str(x), 16) if str(x).startswith('0x') else int(x)).astype('Int64')

    df["ip.src"] = df["ip.src"].fillna("").astype(str).str.strip()
    df["ip.dst"] = df["ip.dst"].fillna("").astype(str).str.strip()
    df["dns.qry.name"] = df["dns.qry.name"].fillna("").astype(str).str.strip()

    startTime = df['frame.time_epoch'].min()
    endTime = df['frame.time_epoch'].max()

    windows = []
    windowID = 0
    time = startTime 

    while time + WINDOW_SIZE <= endTime:
        windowStart = time 
        windowEnd = time + WINDOW_SIZE
        
        window = df[(df['frame.time_epoch'] >= windowStart) &(df['frame.time_epoch'] < windowEnd)]

        if len(window) != 0:

            n = len(window)
            tcpPackets = window[window['ip.proto'] == 6]
            udpPackets = window[window['ip.proto'] == 17]
            tcpCount = len(tcpPackets)
            udpCount = len(udpPackets)

            flags = tcpPackets['tcp.flags'].dropna().astype('int64')
            synCount = int((flags & 0x002).gt(0).sum())
            ackCount = int((flags & 0x010).gt(0).sum())
            finCount = int((flags & 0x001).gt(0).sum())
            rstCount = int((flags & 0x004).gt(0).sum())
            pshCount = int((flags & 0x008).gt(0).sum())
            synOnlyCount = int(((flags & 0x002).gt(0) & (flags & 0x010).eq(0)).sum())

            frameLengths = window['frame.len']
            interArrivalTimes = window['frame.time_delta']

            dns_non_empty = window["dns.qry.name"][window["dns.qry.name"] != ""]

            devicePacketCounts = {}
            for device, ip in DEVICE_IPS.items():
                devicePacketCounts[f'packetTo_{device}'] = int((window['ip.dst'] == ip).sum())
                devicePacketCounts[f'packetFrom_{device}'] = int((window['ip.src'] == ip).sum())

            windows.append({
                'windowID': windowID,
                'windowStart': windowStart,
                'windowEnd': windowEnd,

                'packetCount': n,
                'tcpPacketCount': tcpCount,
                'udpPacketCount': udpCount,
                'tcpRatio' : tcpCount / n,
                'udpRatio' : udpCount / n,

                'avgPacketLength': frameLengths.mean(),
                'stdPacketLength': frameLengths.std(),
                'minPacketLength': frameLengths.min(),
                'maxPacketLength': frameLengths.max(),
                'medianPacketLength': frameLengths.median(),
                'smallPacketCount': (frameLengths < 100).sum(),
                'largePacketCount': (frameLengths > 500).sum(),
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
                'tcpPayloadPacketCount': (tcpPackets['tcp.len'] > 0).sum(),
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

                'packetTo_loungeBulb': devicePacketCounts['packetTo_loungeBulb'],
                'packetFrom_loungeBulb': devicePacketCounts['packetFrom_loungeBulb'],
                'packetTo_bedroomBulb': devicePacketCounts['packetTo_bedroomBulb'],
                'packetFrom_bedroomBulb': devicePacketCounts['packetFrom_bedroomBulb'],
                'packetTo_camera': devicePacketCounts['packetTo_camera'],
                'packetFrom_camera': devicePacketCounts['packetFrom_camera'],
                'packetTo_hub': devicePacketCounts['packetTo_hub'],
                'packetFrom_hub': devicePacketCounts['packetFrom_hub'],
                'packetTo_plug': devicePacketCounts['packetTo_plug'],
                'packetFrom_plug': devicePacketCounts['packetFrom_plug'],

                'tlsHandshakeCount': (window['tls.handshake.type'].dropna() == 1).sum(),
                'avgTLSRecordLen': window['tls.record.length'].dropna().mean(),
                'avgUDPLen': window['udp.length'].dropna().mean(),

                'DNSQueryCount': len(dns_non_empty),
                'uniqueDNSQueries': dns_non_empty.nunique()
            })
        
        time += STEP 
        windowID += 1

    return pd.DataFrame(windows)

def labelWindows(df : pd.DataFrame, activities : list) -> pd.DataFrame:
    def getLabel(row):
        OVERLAP_THRESHOLD = 0.2
        start = row["windowStart"]
        end = row["windowEnd"]
        duration = end - start 

        for activity in activities:
            overlapStart = max(start, activity["start"])
            overlapEnd = min(end, activity["end"])
            overlapDuration = max(0, overlapEnd - overlapStart)

            if overlapDuration / duration >= OVERLAP_THRESHOLD:
                return activity["label"]
        return "idle"
    df["label"] = df.apply(getLabel, axis=1)
    return df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python windowAnalysis2.py <packet csv file> <python object file>")
        sys.exit(1)
    else:
        csvFile = sys.argv[1]
        activityFile = sys.argv[2]

        listOfActivities = loadActivities(activityFile)
        windows = createWindows(csvFile)
        labeledWindows = labelWindows(windows, listOfActivities)

        output = csvFile.replace('.csv', '_windows.csv')
        labeledWindows.to_csv(output, index=False)
        print(f"Saved output to {output}")
        print(labeledWindows.head())

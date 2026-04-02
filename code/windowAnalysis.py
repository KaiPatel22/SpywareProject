import sys 
import random 
import pandas as pd


DEVICE_IPS = {
    'bedroomBulb': '192.168.0.47',
    'loungeBulb': '192.168.0.52',
    'camera': '192.168.0.55',
    'motionSensor': '192.168.0.200',
    'plug': '192.168.0.59',
    'alexa': '192.168.0.35'
}

def loadActivities(filename : str) -> dict:
    listOfActivities = {}
    with open(filename, "r") as f:
        exec(f.read(), listOfActivities)
    return listOfActivities

def loadDf(filename : str) -> pd.DataFrame:
    df = pd.read_csv(filename, escapechar='\\', on_bad_lines='warn', engine='pyarrow').sort_values('frame.time_epoch').reset_index(drop=True) # Testing using pyarrow, according to the documentation it is faster (https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
    print(f"[INFO] Loaded {len(df)} rows from {filename}")
    numericCols = ['ip.ttl', 'ip.len', 'tcp.len' ,'tcp.srcport', 'tcp.dstport', 'tcp.stream', 'tcp.window_size_value', 'tls.handshake.type', 'tls.record.length', 'tcp.analysis.ack_rtt', 'tls.record.content_type', 'udp.srcport', 'udp.dstport']

    for column in numericCols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce') 

    df['tcp.flags'] = df['tcp.flags'].apply(lambda x: int(str(x), 16) if pd.notna(x) and str(x).startswith('0x') else int(x) if pd.notna(x) else 0).astype('int64')
    df['ip.src'] = df['ip.src'].fillna("").astype(str).str.strip()
    df['ip.dst'] = df['ip.dst'].fillna("").astype(str).str.strip()
    df['dns.qry.name'] = df['dns.qry.name'].fillna("").astype(str).str.strip()

    return df

def extractFeatures(windowDF : pd.DataFrame, windowID : int, windowStart : float, windowEnd : float) -> dict:
    WINDOW_SIZE = windowEnd - windowStart
    n = len(windowDF)

    tcpPackets = windowDF[windowDF['ip.proto'] == 6]
    tcpCount = len(tcpPackets)
    udpPackets = windowDF[windowDF['ip.proto'] == 17]
    udpCount = len(udpPackets)

    flags = tcpPackets['tcp.flags']
    synCount = int((flags & 0x002).gt(0).sum())
    ackCount = int((flags & 0x010).gt(0).sum())
    finCount = int((flags & 0x001).gt(0).sum())

    frameLengths = windowDF['frame.len']
    interArrivalTimes = windowDF['frame.time_delta']

    devicePacketCounts = {}
    for device, ip in DEVICE_IPS.items():
        devicePacketCounts[f'packetsTo{device}'] = int((windowDF['ip.dst'] == ip).sum())
        devicePacketCounts[f'packetsFrom{device}'] = int((windowDF['ip.src'] == ip).sum())

    return {
        'windowID' : windowID,
        'windowStart' : windowStart,
        'windowEnd' : windowEnd,

        'packetCount': n,
        'tcpPacketCount' : tcpCount,
        'tcpRatio' : tcpCount / n if n > 0 else 0,
        'udpPacketCount' : udpCount,
        'udpRatio' : udpCount / n if n > 0 else 0,

        'avgPacketLength': frameLengths.mean(),
        'stdPacketLength': frameLengths.std(),
        'minPacketLength': frameLengths.min(),
        'maxPacketLength': frameLengths.max(),
        'medianPacketLength': frameLengths.median(),
        'smallPacketCount': int((frameLengths < 100).sum()),
        'largePacketCount': int((frameLengths > 500).sum()),
        'packetsPerSecond': frameLengths.sum() / WINDOW_SIZE,

        'uniqueSrcIPs': windowDF['ip.src'].nunique(),
        'uniqueDstIPs': windowDF['ip.dst'].nunique(),

        'avgTTL': windowDF['ip.ttl'].mean(),
        'stdTTL': windowDF['ip.ttl'].std(),

        'avgIPLen': windowDF['ip.len'].mean(),
        'stdIPLen': windowDF['ip.len'].std(),

        'uniqueTCPSrcPorts': windowDF['tcp.srcport'].dropna().nunique(),
        'uniqueTCPDstPorts': windowDF['tcp.dstport'].dropna().nunique(),
        'avgTCPLen': tcpPackets['tcp.len'].mean(),
        'stdTCPLen': tcpPackets['tcp.len'].std(),
        'tcpPayloadPacketCount': int((tcpPackets['tcp.len'] > 0).sum()),
        'tcpPayloadPacketRatio': (tcpPackets['tcp.len'] > 0).sum() / tcpCount if tcpCount > 0 else 0,
        'uniqueTCPStreams': windowDF['tcp.stream'].dropna().nunique(),
        'avgTCPWindowSize': windowDF['tcp.window_size_value'].dropna().mean(),
        'minTCPWindowSize': windowDF['tcp.window_size_value'].dropna().min(),
        'maxTCPWindowSize': windowDF['tcp.window_size_value'].dropna().max(),

        'synCount': synCount,
        'ackCount': ackCount,
        'finCount': finCount,

        'uniqueUDPSrcPorts': windowDF['udp.srcport'].dropna().nunique(),
        'uniqueUDPDstPorts': windowDF['udp.dstport'].dropna().nunique(),

        'avgInterArrivalTime': interArrivalTimes.mean(),
        'stdInterArrivalTime': interArrivalTimes.std(),
        'minInterArrivalTime': interArrivalTimes.min(),
        'maxInterArrivalTime': interArrivalTimes.max(),

        **devicePacketCounts,

        'tlsHandshakeCount': (windowDF['tls.handshake.type'].dropna() == 1).sum(),
        'avgTLSRecordLen': windowDF['tls.record.length'].dropna().mean(),
        'stdTLSRecordLen': windowDF['tls.record.length'].dropna().std(),
        'minTLSRecordLen': windowDF['tls.record.length'].dropna().min(),
        'maxTLSRecordLen': windowDF['tls.record.length'].dropna().max(),

        'avgACKRoundTripTime': windowDF['tcp.analysis.ack_rtt'].dropna().mean(),
        'stdACKRoundTripTime': windowDF['tcp.analysis.ack_rtt'].dropna().std(),
        'minACKRoundTripTime': windowDF['tcp.analysis.ack_rtt'].dropna().min(),
        'maxACKRoundTripTime': windowDF['tcp.analysis.ack_rtt'].dropna().max(),
        'ACKRoundTripTimeCount': windowDF['tcp.analysis.ack_rtt'].dropna().count(),

        'avgTimeDelta': tcpPackets['frame.time_delta'].dropna().mean(),
        'stdTimeDelta': tcpPackets['frame.time_delta'].dropna().std(),
        'minTimeDelta': tcpPackets['frame.time_delta'].dropna().min(),
        'maxTimeDelta': tcpPackets['frame.time_delta'].dropna().max(),

        'tlsContentTypeChanegCipherCount': (windowDF['tls.record.content_type'].dropna() == 20).sum(),
        'tlsContentTypeAlertCount': (windowDF['tls.record.content_type'].dropna() == 21).sum(),
        'tlsContentTypeHandshakeCount': (windowDF['tls.record.content_type'].dropna() == 22).sum(),
        'tlsContentTypeAppDataCount': (windowDF['tls.record.content_type'].dropna() == 23).sum(),

    }

def createSlidingWindows(df : pd.DataFrame, activities : dict) -> pd.DataFrame:
    WINDOW_SIZE = 1.0
    STEP = 0.5
    THRESHOLD = 0.5 

    activityList = activities.get('activities', [])
    startTime = df['frame.time_epoch'].min()
    endTime = df['frame.time_epoch'].max()

    rows = []
    windowID = 0 
    time = startTime 

    while time + WINDOW_SIZE <= endTime:
        windowStart = time 
        windowEnd = time + WINDOW_SIZE
        window = df[(df['frame.time_epoch'] >= windowStart) & (df['frame.time_epoch'] < windowEnd)]

        if len(window) > 0:
            print("-" * 20)
            print(f"[ID] : {windowID}")
            row = extractFeatures(window, windowID, windowStart, windowEnd)

            label = "idle"

            for activity in activityList:
                overlapStart = max(windowStart, activity['start'])
                overlapEnd = min(windowEnd, activity['end'])
                overlap = max(0, overlapEnd - overlapStart)
                if overlap / WINDOW_SIZE >= THRESHOLD:
                    label = activity['label']
                    break
            row['label'] = label 
            rows.append(row)
            print(f"[LABEL] : '{label}' and {len(window)} packets")
            print("-" * 20)
            windowID += 1
        time += STEP 

    result = pd.DataFrame(rows).sort_values('windowStart').reset_index(drop=True)
    print(result['label'].value_counts())
    return result

def createEventCenteredWindows(df : pd.DataFrame, activities : dict) -> pd.DataFrame:
    WINDOW_SIZE = 2.0
    activities = activities.get('activities', [])
    rows = []
    windowID = 0

    for activity in activities: 
        center = (activity['start'] + activity['end']) / 2
        windowStart = center - (WINDOW_SIZE / 2)
        windowEnd = center + (WINDOW_SIZE / 2)

        window = df[(df['frame.time_epoch'] >= windowStart) & (df['frame.time_epoch'] < windowEnd)]

        if len(window) > 0:
            print("-" * 20)
            print(f"[ID] : {windowID}")
            row = extractFeatures(window, windowID, windowStart, windowEnd)
            row['label'] = activity['label']
            rows.append(row)
            print(f"[LABEL] : {activity['label']} and {len(window)} packets")
            windowID += 1
    
    result = pd.DataFrame(rows).sort_values('windowStart').reset_index(drop=True)
    print(result['label'].value_counts())
    return result

def createEventCenteredWindowsWithIdle(df : pd.DataFrame, activities : dict) -> pd.DataFrame:
    WINDOW_SIZE = 2.0
    activities = activities.get('activities', [])
    rows = []
    windowID = 0

    for activity in activities: 
        center = (activity['start'] + activity['end']) / 2
        windowStart = center - (WINDOW_SIZE / 2)
        windowEnd = center + (WINDOW_SIZE / 2)

        window = df[(df['frame.time_epoch'] >= windowStart) & (df['frame.time_epoch'] < windowEnd)]

        if len(window) > 0:
            print("-" * 20)
            print(f"[ID] : {windowID}")
            row = extractFeatures(window, windowID, windowStart, windowEnd)
            row['label'] = activity['label']
            rows.append(row)
            print(f"[LABEL] : {activity['label']} and {len(window)} packets")
            windowID += 1
    
    for i in range(len(activities) - 1):
        gapStart = activities[i]['end']
        gapEnd = activities[i + 1]['start']
        print(f"Gap: {gapStart:.2f} -> {gapEnd:.2f} (duration: {gapEnd - gapStart:.2f}s)")
        gapCenter = (gapStart + gapEnd) / 2
        windowStart = gapCenter - (WINDOW_SIZE / 2)
        windowEnd = gapCenter + (WINDOW_SIZE / 2)

        window = df[(df['frame.time_epoch'] >= windowStart) & (df['frame.time_epoch'] < windowEnd)]

        if len(window) > 0:
            print("-" * 20)
            print(f"[ID] : {windowID}")
            row = extractFeatures(window, windowID, windowStart, windowEnd)
            row['label'] = 'idle'
            rows.append(row)
            print(f"[LABEL] : IDLE and {len(window)} packets")
            windowID += 1

    result = pd.DataFrame(rows).sort_values('windowStart').reset_index(drop=True)
    print(result['label'].value_counts())
    return result

def createFullEventWindows(df : pd.DataFrame, activities : dict) -> pd.DataFrame:
    activities = activities.get('activities', [])
    rows = []
    windowID = 0

    for activity in activities: 
        windowStart = activity['start']
        windowEnd = activity['end']

        window = df[(df['frame.time_epoch'] >= windowStart) & (df['frame.time_epoch'] < windowEnd)]

        if len(window) > 0:
            print("-" * 20)
            print(f"[ID] : {windowID}")
            row = extractFeatures(window, windowID, windowStart, windowEnd)
            row['label'] = activity['label']
            rows.append(row)
            print(f"[LABEL] : {activity['label']} and {len(window)} packets")
            windowID += 1
    
    result = pd.DataFrame(rows).sort_values('windowStart').reset_index(drop=True)
    print(result['label'].value_counts())
    return result

def createFullEventWindowsWithIdle(df : pd.DataFrame, activities : dict) -> pd.DataFrame:
    activities = activities.get('activities', [])
    rows = []
    windowID = 0

    for activity in activities: 
        windowStart = activity['start']
        windowEnd = activity['end']

        window = df[(df['frame.time_epoch'] >= windowStart) & (df['frame.time_epoch'] < windowEnd)]

        if len(window) > 0:
            print("-" * 20)
            print(f"[ID] : {windowID}")
            row = extractFeatures(window, windowID, windowStart, windowEnd)
            row['label'] = activity['label']
            rows.append(row)
            print(f"[LABEL] : {activity['label']} and {len(window)} packets")
            windowID += 1
    
    for i in range(len(activities) - 1):
        windowStart = activities[i]['end']
        windowEnd = activities[i + 1]['start']

        window = df[(df['frame.time_epoch'] >= windowStart) & (df['frame.time_epoch'] < windowEnd)]

        if len(window) > 0:
            print("-" * 20)
            print(f"[ID] : {windowID}")
            row = extractFeatures(window, windowID, windowStart, windowEnd)
            row['label'] = 'idle'
            rows.append(row)
            print(f"[LABEL] : IDLE and {len(window)} packets")
            windowID += 1

    result = pd.DataFrame(rows).sort_values('windowStart').reset_index(drop=True)
    print(result['label'].value_counts())
    return result


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python windowAnalysis3.py <csvFile> <activityFile> --flag")
        print(f"{'-' * 20}\n Flag Options:\n --sliding\n --eventCentered\n --eventCenteredWithIdle\n --fullEvent\n --fullEventWithIdle\n{'-' * 20}")
        sys.exit(1)
    
    csvFile = sys.argv[1]
    activityFile = sys.argv[2]
    flag = sys.argv[3]

    activities = loadActivities(activityFile)

    df = loadDf(csvFile)

    if flag == "--sliding":
        windows = createSlidingWindows(df, activities)
    elif flag == "--eventCentered":
        windows = createEventCenteredWindows(df, activities)
    elif flag == "--eventCenteredWithIdle":
        windows = createEventCenteredWindowsWithIdle(df, activities)
    elif flag == "--fullEvent":
        windows = createFullEventWindows(df, activities)
    elif flag == "--fullEventWithIdle":
        windows = createFullEventWindowsWithIdle(df, activities)
    else:
        print(f"Unknown flag '{flag}'. Use --sliding, --eventCentered, --eventCenteredWithIdle, --fullEvent, or --fullEventWithIdle.")
        sys.exit(1)

    windows.to_csv(f"{csvFile.replace('.csv', '_windows.csv')}", index=False)
    print(f"Windowed features saved to {csvFile.replace('.csv', '_windows.csv')}")

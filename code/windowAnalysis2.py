import sys
import pandas as pd

def loadActivities(filename : str) -> dict:
    listOfActivities = {}
    with open(filename, 'r') as f:
        exec(f.read(), listOfActivities)
    return listOfActivities['activities']

def createWindows(filename : str) -> pd.DataFrame:
    WINDOW_SIZE = 5
    STEP = 1

    df = pd.read_csv(filename, escapechar="\\", engine="python", on_bad_lines="warn").sort_values('frame.time_epoch').reset_index(drop=True)

    numericCols = ['frame.time_epoch', 'frame.time_delta', 'frame.len', 'ip.src', 'ip.dst', 'ip.proto', 'ip.ttl', 'ip.len', 'tcp.len', 'tcp.flags', 'tcp.stream' ,'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport']

    for col in numericCols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

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

            dns_non_empty = window["dns.qry.name"][window["dns.qry.name"] != ""]

            windows.append({
                'windowID': windowID,
                'windowStart': windowStart,
                'windowEnd': windowEnd,

                'packetCount': len(window),
                'avgPacketLength': window['frame.len'].mean(),
                'stdPacketLength': window['frame.len'].std(),

                'uniqueSrcIPs': window['ip.src'].nunique(),
                'uniqueDstIPs': window['ip.dst'].nunique(),
                'uniqueTCPSrcPorts': window['tcp.srcport'].dropna().nunique(),
                'uniqueTCPDstPorts': window['tcp.dstport'].dropna().nunique(),
                'uniqueUDPSrcPorts': window['udp.srcport'].dropna().nunique(),
                'uniqueUDPDstPorts': window['udp.dstport'].dropna().nunique(),
                'TCPPacketCount': (window['ip.proto'] == 6).sum(),
                'UDPPacketCount': (window['ip.proto'] == 17).sum(),

                'avgInterArrivalTime': window['frame.time_delta'].mean(),
                'stdInterArrivalTime': window['frame.time_delta'].std(),

                'avgTTL': window['ip.ttl'].mean(),
                'stdTTL': window['ip.ttl'].std(),

                'avgIPLen': window['ip.len'].mean(),
                'stdIPLen': window['ip.len'].std(),
                'avgTCPLen': window['tcp.len'].mean(),
                'stdTCPLen': window['tcp.len'].std(),
                'tcpPayloadPacketCount': (window['tcp.len'] > 0).sum(),

                'uniqueTCPStreams': window['tcp.stream'].dropna().nunique(),

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

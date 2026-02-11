import pandas as pd
import sys 

'''
File used to allocate each packet to a 5 second window. 
This is to be used to analyse and label of windows for idle and events 

We do this because analysing singular packets is not useful as it just shows a snapshot of network traffic at a point in time. 
By grouping the packets we can calculate statistical features such as packet count, average length/size of packets and how many tcp and udp ports
'''

def createWindows(filename : str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    startTime = df['frame.time_epoch'].min()
    df['windowID'] = ((df['frame.time_epoch'] - startTime) // 5).astype(int)

    windowGroupAnalysis = df.groupby('windowID').agg(
        packetCount = ('frame.time_epoch', 'count'),
        avgPacketLength = ('frame.len', 'mean'),
        stdPacketLength = ('frame.len', 'std'),
        uniqueSrcIPs = ('ip.src', 'nunique'),
        uniqueDstIPs = ('ip.dst', 'nunique'),
        uniqueSrcPorts=('tcp.srcport', lambda x: x.dropna().nunique()),
        uniqueDstPorts=('tcp.dstport', lambda x: x.dropna().nunique()),
        tcpPacketCount = ('ip.proto', lambda x: (x == 6).sum()),
        udpPacketCount = ('ip.proto', lambda x: (x == 17).sum())
    )

    max_window = ((df['frame.time_epoch'].max() - startTime) // 5).astype(int)
    all_windows = pd.DataFrame({'windowID': range(max_window + 1)})
    windowGroupAnalysis = all_windows.merge(windowGroupAnalysis, on='windowID', how='left').fillna(0)

    windowGroupAnalysis['windowStart'] = startTime + (windowGroupAnalysis['windowID'] * 5)
    windowGroupAnalysis['windowEnd'] = windowGroupAnalysis['windowStart'] + 5

    columns = windowGroupAnalysis.columns.tolist()
    reordered = ['windowID', 'windowStart', 'windowEnd'] + [col for col in columns if col not in ('windowID', 'windowStart', 'windowEnd')]
    windowGroupAnalysis = windowGroupAnalysis[reordered]

    windowGroupAnalysis.to_csv(f"{filename.replace('.csv', '')}_windows.csv", index=False)
    print(windowGroupAnalysis)

    return windowGroupAnalysis



def labelWindows(dataframe : pd.DataFrame):
    activities = [
        {'start': 1770807493, 'end': 1770807500, 'label': 'alexa_wakeWord'},
        {'start': 1770807551, 'end': 1770807557, 'label': 'alexa_shortResponse'},
        {'start': 1770807690, 'end': 1770807714, 'label': 'alexa_longResponse'},
        {'start': 1770807822, 'end': 1770807845, 'label': 'alexa_requestToPhone'},
        {'start': 1770807894, 'end': 1770807901, 'label': 'alexa_wakeWord'},
        {'start': 1770808017, 'end': 1770808215, 'label': 'alexa_continuousStream'},
        {'start': 1770808410, 'end': 1770808415, 'label': 'alexa_command'},
        {'start': 1770808736, 'end': 1770808744, 'label': 'alexa_shortResponse'},
        {'start': 1770808804, 'end': 1770809057, 'label': 'alexa_continuousStream'},
        {'start': 1770809149, 'end': 1770809156, 'label': 'alexa_wakeWord'},
        {'start': 1770809250, 'end': 1770809260, 'label': 'alexa_mediumResponse'},
        {'start': 1770809358, 'end': 1770809365, 'label': 'alexa_shortResponse'},
        {'start': 1770809441, 'end': 1770809458, 'label': 'alexa_longResponse'},
    ]

    def getLabel(row):
        windowStart = row['windowStart']
        windowEnd = row['windowEnd']
        windowDuration = windowEnd - windowStart

        for activity in activities:
            overlapStart = max(windowStart, activity['start'])
            overlapEnd = min(windowEnd, activity['end'])
            overlapDuration = max(0, overlapEnd - overlapStart)
            if overlapDuration / windowDuration > 0.5:
                return activity['label']
        return 'idle'
    
    dataframe['label'] = dataframe.apply(getLabel, axis = 1)
    dataframe.to_csv(f"{filename.replace('.csv', '')}_labeled.csv", index = False)
    print(dataframe)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python code/windowAnalysis.py <filepath>")
        sys.exit(1)
    else:
        filename = sys.argv[1]

        dataframe = createWindows(filename)
        labelWindows(dataframe)





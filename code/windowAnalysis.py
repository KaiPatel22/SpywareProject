import pandas as pd
import sys 

'''
File used to allocate each packet to a 5 second window. 
This is to be used to analyse and label of windows for idle and events 

We do this because analysing singular packets is not useful as it just shows a snapshot of network traffic at a point in time. 
By grouping the packets we can calculate statistical features such as packet count, average length/size of packets and how many tcp and udp ports
'''





if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python code/windowAnalysis.py <filepath>")
        sys.exit(1)
    else:
        filepath = sys.argv[1]

        df = pd.read_csv(filepath)
        startTime = df['frame.time_epoch'].min()
        df['windowID'] = ((df['frame.time_epoch'] - startTime) // 5).astype(int)

        windowGroupAnalysis = df.groupby('windowID').agg(
            packetCount = ('frame.time_epoch', 'count'),
            avgPacketLength = ('frame.len', 'mean'),
            stdPacketLength = ('frame.len', 'std'),
            uniqueSrcIPs = ('ip.src', 'nunique'),
            uniqueDstIPs = ('ip.dst', 'nunique'),
            uniqueSrcPorts = ('tcp.srcport', 'nunique'),
            uniqueDstPorts = ('tcp.dstport', 'nunique'),
            tcpPacketCount = ('ip.proto', lambda x: (x == 6).sum()),
            udpPacketCount = ('ip.proto', lambda x: (x == 17).sum())
        )
        windowGroupAnalysis['windowStart'] = startTime + (windowGroupAnalysis.index * 5)
        windowGroupAnalysis['windowEnd'] = windowGroupAnalysis['windowStart'] + 5

        cols = windowGroupAnalysis.columns.tolist()
        reordered = ['windowStart', 'windowEnd'] + [c for c in cols if c not in ('windowStart', 'windowEnd')]
        windowGroupAnalysis = windowGroupAnalysis[reordered]

        windowGroupAnalysis.to_csv(f'data/alexaActive_5m_windows.csv')

        print(windowGroupAnalysis)

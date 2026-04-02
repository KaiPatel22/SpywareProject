import pandas as pd
import sys 

'''
File used to allocate each packet to a 5 second window. 
This is to be used to analyse and label of windows for idle and events 

We do this because analysing singular packets is not useful as it just shows a snapshot of network traffic at a point in time. 
By grouping the packets we can calculate statistical features such as packet count, average length/size of packets and how many tcp and udp ports
'''

def loadActivities(filename : str):
    storage = {}
    with open(activityFile, 'r') as file:
        exec(file.read(), storage)
    return storage['activities']

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
    ).reset_index()

    windowGroupAnalysis['windowStart'] = startTime + (windowGroupAnalysis['windowID'] * 5)
    windowGroupAnalysis['windowEnd'] = windowGroupAnalysis['windowStart'] + 5

    columns = windowGroupAnalysis.columns.tolist()
    reordered = ['windowID', 'windowStart', 'windowEnd'] + [col for col in columns if col not in ('windowID', 'windowStart', 'windowEnd')]
    windowGroupAnalysis = windowGroupAnalysis[reordered]

    # windowGroupAnalysis.to_csv(f"{filename.replace('.csv', '')}_windows.csv", index=False)
    print(windowGroupAnalysis)

    return windowGroupAnalysis



def labelWindows(dataframe : pd.DataFrame):
    # activities = [
    #     {'start': 1770807493, 'end': 1770807500, 'label': 'alexa_wakeWord'},
    #     {'start': 1770807551, 'end': 1770807557, 'label': 'alexa_shortResponse'},
    #     {'start': 1770807690, 'end': 1770807714, 'label': 'alexa_longResponse'},
    #     {'start': 1770807894, 'end': 1770807901, 'label': 'alexa_wakeWord'},
    #     {'start': 1770808017, 'end': 1770808215, 'label': 'alexa_continuousStream'},
    #     {'start': 1770808410, 'end': 1770808415, 'label': 'alexa_command'},
    #     {'start': 1770808736, 'end': 1770808744, 'label': 'alexa_shortResponse'},
    #     {'start': 1770808804, 'end': 1770809057, 'label': 'alexa_continuousStream'},
    #     {'start': 1770809149, 'end': 1770809156, 'label': 'alexa_wakeWord'},
    #     {'start': 1770809250, 'end': 1770809260, 'label': 'alexa_mediumResponse'},
    #     {'start': 1770809358, 'end': 1770809365, 'label': 'alexa_shortResponse'},
    #     {'start': 1770809441, 'end': 1770809458, 'label': 'alexa_longResponse'},
    #     {'start': 1770907231, 'end': 1770907239, 'label': 'alexa_wakeWord'},
    #     {'start': 1770907271, 'end': 1770907275, 'label': 'alexa_command'},
    #     {'start': 1770907312, 'end': 1770907330, 'label': 'alexa_longResponse'},
    #     {'start': 1770907390, 'end': 1770907397, 'label': 'alexa_shortResponse'},
    #     {'start': 1770907421, 'end': 1770907429, 'label': 'alexa_wakeWord'},
    #     {'start': 1770907458, 'end': 1770907582, 'label': 'alexa_continuousStream'},
    #     {'start': 1770907596, 'end': 1770907602, 'label': 'alexa_wakeWord'},
    #     {'start': 1770907659, 'end': 1770907669, 'label': 'alexa_mediumResponse'},
    #     {'start': 1770907697, 'end': 1770907703, 'label': 'alexa_shortResponse'},
    #     {'start': 1770907726, 'end': 1770907742, 'label': 'alexa_longResponse'},
    #     {'start': 1770907775, 'end': 1770907782, 'label': 'alexa_shortResponse'},
    #     {'start': 1770907838, 'end': 1770907846, 'label': 'alexa_wakeWord'},
    #     {'start': 1770907870, 'end': 1770907879, 'label': 'alexa_mediumResponse'},
    #     {'start': 1770907945, 'end': 1770907961, 'label': 'alexa_longResponse'},
    #     {'start': 1770908008, 'end': 1770908030, 'label': 'alexa_longResponse'},
    #     {'start': 1770908074, 'end': 1770908078, 'label': 'alexa_command'},
    #     {'start': 1770908129, 'end': 1770908133, 'label': 'alexa_command'},
    #     {'start': 1770908159, 'end': 1770908166, 'label': 'alexa_wakeWord'},
    #     {'start': 1770908185, 'end': 1770908196, 'label': 'alexa_shortResponse'},
    #     {'start': 1770908209, 'end': 1770908214, 'label': 'alexa_command'},
    #     {'start': 1770908229, 'end': 1770908365, 'label': 'alexa_continuousStream'},
    #     {'start': 1770908399, 'end': 1770908404, 'label': 'alexa_shortResponse'},
    #     {'start': 1770908480, 'end': 1770908487, 'label': 'alexa_wakeWord'},
    # ]

    # activities = [
    #     {"start": 1771600810, "end": 1771600816, "label":"bulbOn_viaPhone"},
    #     {"start": 1771600841, "end": 1771600845, "label":"bulbOn_viaPhone"},
    #     {"start": 1771600876, "end": 1771600881, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771600876, "end": 1771600881, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771600912, "end": 1771600917, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771600951, "end": 1771600961, "label":"phoneApp_noActivity"},
    #     {"start": 1771600977, "end": 1771600983, "label":"bulbChange_viaPhone"},
    #     {"start": 1771601008, "end": 1771601012, "label":"BulbOff_viaAlexa"},
    #     {"start": 1771601044, "end": 1771601057, "label":"phoneApp_noActivity"},
    #     {"start": 1771601084, "end": 1771601089, "label":"bulbOn_viaPhone"},
    #     {"start": 1771601133, "end": 1771601139, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771601167, "end": 1771601175, "label":"bulbChange_viaPhone"},
    #     {"start": 1771601210, "end": 1771601216, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771601250, "end": 1771601257, "label":"phoneApp_noActivity"},
    #     {"start": 1771601293, "end": 1771601300, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771601373, "end": 1771601378, "label":"bulbChange_viaPhone"},
    #     {"start": 1771601403, "end": 1771601414, "label":"bulbChange_viaPhone"},
    #     {"start": 1771601451, "end": 1771601457, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771601482, "end": 1771601487, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771601511, "end": 1771601527, "label":"phoneApp_noActivity"},
    #     {"start": 1771601559, "end": 1771601567, "label":"bulbChange_viaPhone"},
    #     {"start": 1771601659, "end": 1771601670, "label":"bulbChange_viaPhone"},
    #     {"start": 1771601702, "end": 1771601709, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771601724, "end": 1771601729, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771601749, "end": 1771601754, "label":"phoneApp_noActivity"},
    #     {"start": 1771601793, "end": 1771601799, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771601818, "end": 1771601824, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771601842, "end": 1771601846, "label":"bulbOff_viaPhone"},
    #     {"start": 1771601902, "end": 1771601906, "label":"bulbOn_viaAlexa"},
    #     {"start": 1771601937, "end": 1771601946, "label":"bulbChange_viaPhone"},
    #     {"start": 1771602029, "end": 1771602034, "label":"bulbOff_viaAlexa"},
    #     {"start": 1771602054, "end": 1771602060, "label":"bulbOn_viaPhone"},
    #     {"start": 1771602097, "end": 1771602101, "label":"bulbOff_viaAlexa"},
    #     {"start": 1771602119, "end": 1771602127, "label":"phoneApp_noActivity"},
    #     {"start": 1771602159, "end": 1771602165, "label":"bulbChange_viaAlexa"},
    #     {"start": 1771602188, "end": 1771602193, "label":"bulbOn_viaPhone"},
    #     {"start": 1771602217, "end": 1771602221, "label":"bulbOn_viaAlexa"},
    #     {"start": 1771602271, "end": 1771602275, "label":"bulbOff_viaAlexa"},
    #     {"start": 1771602291, "end": 1771602298, "label":"bulbOff_viaPhone"},
    # ]

    # activities = [
    #     {"start": 1771866243, "end": 1771866261, "label": "exithome"},
    #     {"start": 1771866296, "end": 1771866313, "label": "enterhome"},
    #     {"start": 1771866337, "end": 1771866354, "label": "exithome"},
    #     {"start": 1771866378, "end": 1771866394, "label": "enterhome"},
    #     {"start": 1771866416, "end": 1771866432, "label": "exithome"},
    #     {"start": 1771866449, "end": 1771866467, "label": "enterhome"},
    #     {"start": 1771866479, "end": 1771866496, "label": "exithome"},
    #     {"start": 1771866512, "end": 1771866529, "label": "enterhome"},
    #     {"start": 1771866551, "end": 1771866573, "label": "exithome"},
    #     {"start": 1771866583, "end": 1771866600, "label": "enterhome"},
    #     {"start": 1771866611, "end": 1771866627, "label": "exithome"},
    #     {"start": 1771866635, "end": 1771866651, "label": "enterhome"},
    #     {"start": 1771866658, "end": 1771866672, "label": "exithome"},
    #     {"start": 1771866688, "end": 1771866700, "label": "enterhome"},
    #     {"start": 1771866713, "end": 1771866725, "label": "exithome"},
    #     {"start": 1771866743, "end": 1771866753, "label": "enterhome"},
    #     {"start": 1771866777, "end": 1771866794, "label": "exithome"},
    #     {"start": 1771866805, "end": 1771866819, "label": "enterhome"},
    #     {"start": 1771866832, "end": 1771866849, "label": "exithome"},
    #     {"start": 1771866859, "end": 1771866874, "label": "enterhome"},
    #     {"start": 1771866887, "end": 1771866901, "label": "exithome"},
    #     {"start": 1771866915, "end": 1771866928, "label": "enterhome"},
    #     {"start": 1771866945, "end": 1771866960, "label": "exithome"},
    #     {"start": 1771866969, "end": 1771866981, "label": "enterhome"},
    #     {"start": 1771866997, "end": 1771867011, "label": "exithome"},
    #     {"start": 1771867036, "end": 1771867049, "label": "enterhome"},
    #     {"start": 1771867060, "end": 1771867077, "label": "exithome"},
    #     {"start": 1771867089, "end": 1771867101, "label": "enterhome"},
    #     {"start": 1771867129, "end": 1771867143, "label": "exithome"},
    #     {"start": 1771867161, "end": 1771867175, "label": "enterhome"},
    # ]

    # activities = [
    #     {"start": 1772472833, "end": 1772472857, "label": "exithome"},
    #     {"start": 1772472867, "end": 1772472887, "label": "enterhome"},
    #     {"start": 1772472898, "end": 1772472917, "label": "exithome"},
    #     {"start": 1772472925, "end": 1772472937, "label": "enterhome"},
    #     {"start": 1772472961, "end": 1772472975, "label": "exithome"},
    #     {"start": 1772472987, "end": 1772473000, "label": "enterhome"},
    #     {"start": 1772473013, "end": 1772473037, "label": "exithome"},
    #     {"start": 1772473046, "end": 1772473060, "label": "enterhome"},
    #     {"start": 1772473072, "end": 1772473100, "label": "onphoneapp"},
    #     {"start": 1772473109, "end": 1772473122, "label": "exithome"},
    #     {"start": 1772473166, "end": 1772473186, "label": "exithome"},
    #     {"start": 1772473257, "end": 1772473296, "label": "enterhome"},
    #     {"start": 1772473296, "end": 1772473296, "label": "exithome"},
    #     {"start": 1772473400, "end": 1772473428, "label": "enterhome"},
    #     {"start": 1772473457, "end": 1772473490, "label": "onphoneapp"},
    #     {"start": 1772473519, "end": 1772473548, "label": "exithome"},
    #     {"start": 1772473562, "end": 1772473576, "label": "enterhome"},
    #     {"start": 1772473587, "end": 1772473606, "label": "exithome"},
    # ]

    # activities = [
    #     {"start": 1772619509, "end": 1772619513, "label": "bulboff"},
    #     {"start": 1772619547, "end": 1772619551, "label": "bulbon"},
    #     {"start": 1772619600, "end": 1772619604, "label": "bulbchange"},
    #     {"start": 1772619655, "end": 1772619659, "label": "bulbchange"},
    #     {"start": 1771848836, "end": 1771848837, "label": "movingaroundhouse"},
    #     {"start": 1772619707, "end": 1772619711, "label": "bulbchange"},
    #     {"start": 1771848836, "end": 1771848837, "label": "movingaroundhouse"},
    #     {"start": 1772620191, "end": 1772620195, "label": "bulbchange"},
    #     {"start": 1771848836, "end": 1771848837, "label": "movingaroundhouse"},
    # ]

    def getLabel(row):
        windowStart = row['windowStart']
        windowEnd = row['windowEnd']
        windowDuration = windowEnd - windowStart

        for activity in activities:
            overlapStart = max(windowStart, activity['start'])
            overlapEnd = min(windowEnd, activity['end'])
            overlapDuration = max(0, overlapEnd - overlapStart)
            if overlapDuration / windowDuration >= 0.2:
                return activity['label']
        return 'idle'
    
    dataframe['label'] = dataframe.apply(getLabel, axis = 1)
    dataframe.to_csv(f"{filename.replace('.csv', '')}_labeled.csv", index = False)
    print(dataframe)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python code/windowAnalysis.py <csv for packets> <py of activities>")
        sys.exit(1)
    else:
        filename = sys.argv[1]
        activityFile = sys.argv[2]

        activities = loadActivities(activityFile)
        # print(activities)
        dataframe = createWindows(filename)
        labelWindows(dataframe)





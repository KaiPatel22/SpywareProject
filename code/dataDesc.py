import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/Users/kaipatel/Documents/SpywareProject/data/homeAll_extracted_windows.csv")

print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns)
print("DataFrame info:")
print(df.info())
print("DataFrame head:")
print(df.head())

print(f" Value counts for each class: {df['label'].value_counts()}")


requested_cols = ['windowID','windowStart','windowEnd','packetCount','tcpPacketCount','udpPacketCount','tcpRatio','udpRatio','avgPacketLength','stdPacketLength','minPacketLength','maxPacketLength','medianPacketLength','smallPacketCount','largePacketCount','throughput','uniqueSrcIPs','uniqueDstIPs','avgTTL','stdTTL','avgIPLen','stdIPLen','uniqueTCPSrcPorts','uniqueTCPDstPorts','avgTCPLen','stdTCPLen','tcpPayloadPacketCount','tcpPayloadPacketRatio','uniqueTCPStreams','avgTCPWindowSize','minTCPWindowSize','tcpSynCount','tcpAckCount','tcpFinCount','tcpRstCount','tcpPshCount','tcpSynOnlyCount','uniqueUDPSrcPorts','uniqueUDPDstPorts','avgInterArrivalTime','stdInterArrivalTime','minInterArrivalTime','maxInterArrivalTime','packetTo_loungeBulb','packetFrom_loungeBulb','packetTo_bedroomBulb','packetFrom_bedroomBulb','packetTo_camera','packetFrom_camera','packetTo_hub','packetFrom_hub','packetTo_plug','packetFrom_plug','tlsHandshakeCount','avgTLSRecordLen','avgUDPLen','DNSQueryCount','uniqueDNSQueries','label']


numeric_cols = [c for c in requested_cols if c in df.columns]
missing_cols = [c for c in requested_cols if c not in df.columns]  # fix

if missing_cols:
    print(f"Skipping missing columns: {missing_cols}")

# Remove constant/all-null columns so correlation is defined
usable_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
dropped_cols = [c for c in numeric_cols if c not in usable_cols]
if dropped_cols:
    print(f"Dropping constant/unusable columns: {dropped_cols}")

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'].astype(str))

print("Label mapping:")
for i, cls in enumerate(le.classes_):
    print(f"{i} -> {cls}")

corr = (
    df[usable_cols + ['label_encoded']]
    .corr(method='spearman', numeric_only=True)['label_encoded']
    .drop('label_encoded')
    .dropna()
)

corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
print(corr)

print(df.groupby('label')[['packetCount','avgPacketLength',
    'avgInterArrivalTime','tcpPacketCount','udpPacketCount']].mean())
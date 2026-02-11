import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('data/alexaMixedTest_windows.csv')

features = [
    'packetCount',
    'avgPacketLength',
    'stdPacketLength',
    'uniqueSrcIPs',
    'uniqueDstIPs',
    'uniqueSrcPorts',
    'uniqueDstPorts',
    'tcpPacketCount',
    'udpPacketCount'
]
X = df[features]

X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Print cluster statistics
cluster_stats = df.groupby('cluster')[features].mean()
print("\nCluster Statistics:")
print(cluster_stats)

# Optional: Print cluster sizes
print("\nCluster Sizes:")
print(df['cluster'].value_counts().sort_index())
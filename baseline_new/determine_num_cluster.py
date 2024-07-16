import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
import numpy as np
import os, torch
from tqdm import tqdm

pwd=os.getcwd()
print("Start Loading Data")
X = torch.tensor(torch.load(f'{pwd}/data_new/new_train_x.pth')).float()[0]
print("Finish Loading Data")

# Determine the WCSS for different number of clusters
wcss = []
silhouette_scores = []
davies_bouldin_scores = []

for i in tqdm(range(2, 30)):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    davies_bouldin_avg = davies_bouldin_score(X, cluster_labels)
    davies_bouldin_scores.append(davies_bouldin_avg)

# Plotting the results
fig, ax1 = plt.subplots(figsize=(15, 12))

ax1.plot(range(2, 30), wcss, 'bo-', label='WCSS')
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('WCSS')

ax2 = ax1.twinx()
ax2.plot(range(2, 30), silhouette_scores, 'ro-', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score')

ax3 = ax1.twinx()
ax3.plot(range(2, 30), davies_bouldin_scores, 'go-', label='Davies-Bouldin Index')

fig.legend(loc='upper right', bbox_to_anchor=(1.15, 1), bbox_transform=ax1.transAxes)

plt.title('Clustering Metrics')
plt.show()
plt.savefig("baseline_new/cluster_statistics.png")
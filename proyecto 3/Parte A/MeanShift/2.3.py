import numpy as np
import matplotlib.pyplot as plt

X = np.load("MeanShift/X.npy")
y = np.load("MeanShift/_ .npy")
from sklearn.cluster import MeanShift, estimate_bandwidth

# MeanShift
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_
centers = ms.cluster_centers_
n_clusters = len(np.unique(labels))

print("Número de clusters:", n_clusters)
print("Centroides:\n", centers)

# Figura C
plt.figure(figsize=(6,4))
plt.scatter(X[:,0], X[:,1], c=labels, s=12, cmap="viridis")
plt.scatter(centers[:,0], centers[:,1], c="black", s=250, marker="*")
plt.title("Figura C - MeanShift")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

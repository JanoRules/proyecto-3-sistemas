import numpy as np
import matplotlib.pyplot as plt

X = np.load("MeanShift/X.npy")     # ajusta la ruta si hace falta
y = np.load("MeanShift/_ .npy")
from sklearn.cluster import MeanShift, estimate_bandwidth

# estimar bandwidth automáticamente
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500, random_state=0)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_               # etiquetas predichas por MeanShift
centers = ms.cluster_centers_     # centroides
n_clusters = len(np.unique(labels))

print("N° clusters:", n_clusters)
print("Centroides:\n", centers)

# Figura C: clusters coloreados + centroides con estrella negra
plt.figure(figsize=(6,4))
plt.scatter(X[:,0], X[:,1], c=labels, s=12, cmap="viridis")
plt.scatter(centers[:,0], centers[:,1], c="black", s=250, marker="*")
plt.title("Figura C (MeanShift)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

unique_labels, counts = np.unique(y, return_counts=True)

print("Etiquetas reales:", unique_labels)
print("Cantidad por etiqueta:", counts)
print("Total de datos:", counts.sum())

test = np.array([
    [-7, -6],
    [1.5, -6.5],
    [7.9, 0.5],
    [5.5, 10]
])

test_labels = ms.predict(test)
print("Predicción de MeanShift:", test_labels)

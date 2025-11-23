import numpy as np
import matplotlib.pyplot as plt

X = np.load("MeanShift/X.npy")
y = np.load("MeanShift/_ .npy")
from sklearn.cluster import MeanShift, estimate_bandwidth

# Ajustar modelo
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

test = np.array([
    [-7, -6],
    [1.5, -6.5],
    [7.9, 0.5],
    [5.5, 10]
])

test_labels = ms.predict(test)
print("Clusters predichos:", test_labels)

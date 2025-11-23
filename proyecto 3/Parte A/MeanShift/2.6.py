import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = np.load("MeanShift/X.npy")
y = np.load("MeanShift/_ .npy")
from sklearn.cluster import MeanShift, estimate_bandwidth

# Entrenar MeanShift
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_

# Mapeo cluster → clase real
cont = pd.crosstab(y, labels)
mapping = cont.idxmax(axis=0).to_dict()

# Test set
test = np.array([
    [-7, -6],
    [1.5, -6.5],
    [7.9, 0.5],
    [5.5, 10]
])

test_labels = ms.predict(test)
real_classes = [mapping[l] for l in test_labels]

print("Clusters asignados:", test_labels)
print("Clases reales correspondientes:", real_classes)

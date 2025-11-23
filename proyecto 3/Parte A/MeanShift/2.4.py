import numpy as np
import matplotlib.pyplot as plt

X = np.load("MeanShift/X.npy")
y = np.load("MeanShift/_ .npy")


unique_labels, counts = np.unique(y, return_counts=True)

print("Etiquetas reales:", unique_labels)
print("Cantidad por etiqueta:", counts)
print("Cantidad total:", counts.sum())

import numpy as np
import matplotlib.pyplot as plt

X = np.load("MeanShift/X.npy")     # ajusta la ruta si hace falta
y = np.load("MeanShift/_ .npy")    # nombre raro pero válido

plt.figure(figsize=(6,4))
plt.scatter(X[:,0], X[:,1], s=12)
plt.title("Figura A")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

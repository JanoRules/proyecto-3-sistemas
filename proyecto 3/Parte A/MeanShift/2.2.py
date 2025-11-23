import numpy as np
import matplotlib.pyplot as plt

X = np.load("MeanShift/X.npy")
y = np.load("MeanShift/_ .npy")


# Figura B
plt.figure(figsize=(6,4))
plt.scatter(X[:,0], X[:,1], c=y, s=12, cmap="viridis")
plt.title("Figura B")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

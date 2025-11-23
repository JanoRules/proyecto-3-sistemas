import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np

# Cargar el conjunto de datos de la flor Iris
iris = load_iris()

# Seleccionar solo las dos características (columnas) para la visualización en 2D
# Columna 0: Longitud del Sépalo (Sepal Length)
# Columna 1: Ancho del Sépalo (Sepal Width)
# X contiene todas las 150 muestras
X = iris.data[:, :2]

# 3 clusteres = tres grupos a graficar
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X)

# Guardamos las etiquetas de cluster originales (0, 1, 2)
original_labels = kmeans.labels_
new_labels = np.copy(original_labels)

# --- Intercambio de Colores ---
# Usamos el valor '99' como un marcador temporal para evitar sobrescribir
# los grupos durante el intercambio de etiquetas 0 y 1.

# Paso 1: Mover temporalmente el Cluster 0.
# Los puntos que originalmente son '0' ahora se marcan como '99'.
new_labels[original_labels == 0] = 99 
new_labels[original_labels == 1] = 0
new_labels[new_labels == 99] = 1

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=new_labels, cmap='viridis', s=50) 
plt.xlabel('Longitud del Sépalo (cm)')
plt.ylabel('Ancho del Sépalo (cm)')
plt.title('Gráfico de Clusters del Dataset Iris (Colores Intercambiados)')
plt.show()
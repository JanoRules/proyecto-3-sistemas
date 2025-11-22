import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans 
import os

# --- 1. Definición de Rutas y Parámetros ---

DATA_PATH = os.path.join('KMeans', 'A.npy')

N_CLUSTERS = 3 

# Definimos la paleta de colores requerida (Verde, Rojo, Azul)
colors = ['green', 'red', 'blue']
custom_cmap = ListedColormap(colors)

# --- 2. Carga y Aplicación de K-Means ---
try:
    # Cargar los datos de entrada (Características X)
    X = np.load(DATA_PATH)
    
except FileNotFoundError as e:
    print(f"❌ Error al cargar el archivo: {e}")
    print("Asegúrate de que el archivo 'A.npy' esté en la carpeta 'KMeans'.")
    exit()

# Aplicar el Algoritmo K-Means (sklearn)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# --- 4. RESPUESTA A LAS PREGUNTAS (Nueva Sección) ---

print("\n----------------------------------------------------")
print("4. ¿Cuáles son las etiquetas de la data?¿Cuántos datas son?")
print("----------------------------------------------------")
print(f"Número total de datos (muestras): {X.shape[0]}")
print(f"Número de características por dato: {X.shape[1]}")
print(f"Las etiquetas de clústeres obtenidas son: {np.unique(labels)}")
print(f"Número de clústeres identificados: {len(np.unique(labels))}")
print("----------------------------------------------------")


# --- 5. Visualización de la Figura B con Centroides ---
    
# Separamos las dos primeras características para el eje X y el eje Y
x_coords = X[:, 0]
y_coords = X[:, 1]

plt.figure(figsize=(8, 5))

# 5.1 Graficar los puntos (Clústeres Coloreados)
scatter = plt.scatter(
    x_coords, 
    y_coords, 
    c=labels,              
    cmap=custom_cmap,      
    s=35,
    alpha=1.0              
)

# 5.2 Graficar los Centroides (Estrella Negra)
plt.scatter(
    centroids[:, 0],       
    centroids[:, 1],       
    marker='*',            
    s=300,                 
    color='black',         
    edgecolors='white',    
    label='Centroides'     
)

# Añadir títulos y etiquetas
plt.title('Visualización de Clústeres K-Means y Centroides')
plt.xlabel('Feature A')
plt.ylabel('Feature B')

# Añadir la leyenda de los clústeres
legend1 = plt.legend(*scatter.legend_elements(), 
                     loc="lower left", 
                     title="Clústeres")
plt.gca().add_artist(legend1) 

# Añadir la leyenda de los centroides
plt.legend(loc='upper right')

plt.show()
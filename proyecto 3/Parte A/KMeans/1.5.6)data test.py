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


# --- 3. RESPUESTA A LA PREGUNTA 4 (Estadísticas) ---

print("\n----------------------------------------------------")
print("4. Análisis de la Data y Resultados del K-Means")
print("----------------------------------------------------")
print(f"Número total de datos (muestras): {X.shape[0]}")
print(f"Número de características por dato: {X.shape[1]}")
print(f"Las etiquetas de clústeres obtenidas son: {np.unique(labels)}")
print(f"Número de clústeres identificados: {len(np.unique(labels))}")
print("----------------------------------------------------")


# --- 4. PREPARACIÓN DE DATOS DE PRUEBA ---

data_test = np.array([ 
    [2.0, 5.0],
    [3.2, 6.5],
    [7.0, 2.5],
    [9.0, 3.2],
    [9.0, -6.0],
    [11.0, -8.0]
])

# --- 5. PREDICCIÓN (Preguntas 5 y 6) ---

# 5.1 Realizar la predicción
predictions = kmeans.predict(data_test)

# 5.2 Imprimir la tabla de resultados
print("\n----------------------------------------------------")
print("5. y 6. Predicción del Data Test (Asignación de Clase)")
print("----------------------------------------------------")
print("{:<20} | {:<20}".format("Data Test", "Clase (Cluster)"))
print("-" * 43)

for data, cluster in zip(data_test, predictions):
    # Formateamos el array para que se imprima limpio
    data_str = str(data.round(1)).replace('\n', '') 
    print("{:<20} | {:<20}".format(data_str, cluster))
print("----------------------------------------------------")


# --- 6. Visualización de la Figura B con Centroides (Gráfico) ---
    
# Separamos las dos primeras características para el eje X y el eje Y
x_coords = X[:, 0]
y_coords = X[:, 1]

plt.figure(figsize=(8, 5))

# 6.1 Graficar los puntos (Clústeres Coloreados)
scatter = plt.scatter(
    x_coords, 
    y_coords, 
    c=labels,              
    cmap=custom_cmap,      
    s=35,
    alpha=1.0              
)

# 6.2 Graficar los Centroides (Estrella Negra)
plt.scatter(
    centroids[:, 0],       
    centroids[:, 1],       
    marker='*',            
    s=300,                 
    color='black',         
    edgecolors='white',    
    label='Centroides'     
)

# 6.3 Graficar los Data Test (Opcional, pero útil para visualización)
plt.scatter(
    data_test[:, 0],
    data_test[:, 1],
    marker='D',            # Marcador de diamante para el test
    s=80,
    color='magenta',
    edgecolors='black',
    label='Data Test'
)

# Añadir títulos y etiquetas
plt.title('Visualización de Clústeres K-Means, Centroides y Data Test')
plt.xlabel('Feature A')
plt.ylabel('Feature B')

# Añadir las leyendas
legend1 = plt.legend(*scatter.legend_elements(), 
                     loc="lower left", 
                     title="Clústeres")
plt.gca().add_artist(legend1) 

plt.legend(loc='upper right')

plt.show()
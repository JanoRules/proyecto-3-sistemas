import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap # Necesario para definir un mapa de colores personalizado
import os

# --- 1. Definición de Rutas de Archivos ---

DATA_PATH = os.path.join('KMeans', 'A.npy')
LABELS_PATH = os.path.join('KMeans', '_.npy')

# --- 2. Carga de los Archivos .npy ---
try:
    X = np.load(DATA_PATH)      # Datos de entrada
    labels = np.load(LABELS_PATH)  # Etiquetas de clúster
    
except FileNotFoundError as e:
    print(f"❌ Error al cargar el archivo: {e}")
    print("Asegúrate de que los archivos 'A.npy' y '_.npy' estén en la carpeta 'KMeans'.")
    exit()

# --- 3. Verificación de la Estructura de Datos ---
print(f"Datos (X) cargados. Forma: {X.shape}")
print(f"Etiquetas (labels) cargadas. Forma: {labels.shape}")

# El resto de las validaciones de dimensiones se mantienen...

# Separamos las dos primeras características para el eje X y el eje Y
x_coords = X[:, 0]
y_coords = X[:, 1]

# --- 4. Definición del Mapa de Colores Específico (Figura B) ---

# Definimos la paleta de colores requerida (Verde, Rojo, Azul)
# Esto mapeará: 
# Etiqueta 0 -> Verde
# Etiqueta 1 -> Rojo
# Etiqueta 2 -> Azul
colors = ['green', 'red', 'blue']
custom_cmap = ListedColormap(colors)

# --- 5. Visualización de la Figura B ---
    
plt.figure(figsize=(8, 5))

# Graficamos, utilizando el mapa de colores personalizado 'custom_cmap'
scatter = plt.scatter(
    x_coords, 
    y_coords, 
    c=labels,          # Colores basados en las etiquetas
    cmap=custom_cmap,  # ¡USAMOS EL MAPA PERSONALIZADO!
    s=35,              # Tamaño de los puntos ajustado para Figura B
    alpha=1.0          # Opacidad total (1.0) para que los colores sean sólidos como en la referencia
)

# Añadir títulos y etiquetas de los ejes
plt.title('Visualización de Figura B')
plt.xlabel('Feature A')
plt.ylabel('Feature B')

# Añadir una leyenda para los clústeres
legend1 = plt.legend(*scatter.legend_elements(), 
                     loc="lower left", 
                     title="Clústeres")
plt.gca().add_artist(legend1)

# Mostrar el gráfico (sin cuadrícula para ser más fiel a la Figura B)
plt.show()
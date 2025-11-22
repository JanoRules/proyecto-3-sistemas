import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Rutas de archivo ---
DATA_PATH_A = os.path.join('KMeans', 'A.npy')
LABELS_PATH = os.path.join('KMeans', '_.npy') # Aún se carga para cumplir con la instrucción de usar ambos archivos

# --- 2. Carga de los archivos ---
try:
    # X: Datos de entrada
    X = np.load(DATA_PATH_A)      
    # labels: Etiquetas de clúster (Se cargan pero no se usan en el gráfico)
    labels = np.load(LABELS_PATH)  
except FileNotFoundError as e:
    print(f"❌ Error al cargar archivo: {e}")
    exit()

# --- 3. Validación y Estructura ---
print(f"A.npy → Datos cargados (X.shape): {X.shape}")
print(f"_.npy → Etiquetas cargadas (labels.shape): {labels.shape}")

if X.shape[0] != labels.shape[0]:
    print("❌ Error: A.npy y _.npy no tienen el mismo número de muestras.")
    exit()

# Preparar puntos para graficar (las dos características)
x_coords = X[:, 0]
y_coords = X[:, 1]

# --- 4. Gráfico Final (Un Solo Color) ---
plt.figure(figsize=(8, 5)) # Se ajusta el tamaño para ser similar a la figura de referencia

# La clave: El color es fijo ('#1f77b4' es el azul por defecto de Matplotlib)
plt.scatter(
    x_coords,
    y_coords,
    s=35,               # Tamaño de los puntos
    color='#1f77b4',    # Un color fijo para todos los puntos
    alpha=0.8           # Transparencia
)

# Añadir títulos y etiquetas de los ejes como en la Figura A de referencia
plt.title("Visualización de Datos Crudos")
plt.xlabel("Feature A")
plt.ylabel("Feature B")

# IMPORTANTE: Se elimina la leyenda de clústeres para que todos los puntos sean iguales.
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
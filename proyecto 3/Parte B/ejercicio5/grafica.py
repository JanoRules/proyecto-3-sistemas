import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d3 = np.load('ejercicio5\puntos_3d.npy')

# Crear el DataFrame
df = pd.DataFrame(d3)

# Asignar nombres a las columnas
# Completamos: df.columns = [....]
df.columns = ['x', 'y', 'z']

# El centroide es el promedio de cada columna
centroide = df.mean() 

print("Coordenadas del Centroide:")
print(centroide)


fig = plt.figure(figsize=(10, 5))

# Gráfico 1: Sin Centroide
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(df['x'], df['y'], df['z'], c='red', alpha=0.6)
ax1.set_title("Sin Centroide")
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

# Gráfico 2: Con Centroide (El que pide el ejercicio)
ax2 = fig.add_subplot(122, projection='3d')

# a) Graficamos los puntos rojos
ax2.scatter(df['x'], df['y'], df['z'], c='red', alpha=0.6, label='Datos')

# Grafico CENTROIDE AZUL ; s=x es el tamaño
ax2.scatter(centroide['x'], centroide['y'], centroide['z'], 
            c='blue', s=200, marker='o', label='Centroide')

ax2.set_title("Con Centroide")
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.legend()

plt.tight_layout()
plt.show()
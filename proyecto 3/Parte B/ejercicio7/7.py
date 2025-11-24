# ===========================================
# Filtro de Kalman para predecir posición
# ===========================================

import numpy as np
import matplotlib.pyplot as plt

# ===========================================
# 1) Trayectoria real de la partícula
# ===========================================
dt = 0.1
t = np.arange(0, 100, dt)
pos = 0.1*(3*t - t**2)   # trayectoria real

# ===========================================
# 2) Simular mediciones ruidosas del sensor
# ===========================================
sigma_z = 20  # ruido del sensor
z = pos + np.random.normal(0, sigma_z, len(pos))

# ===========================================
# 3) Filtro de Kalman (modelo posición–velocidad)
# ===========================================

# Matriz de transición de estados
F = np.array([
    [1, dt],
    [0, 1]
])

# Matriz de observación
H = np.array([[1, 0]])

# Ruido del proceso (asumiendo aceleración desconocida)
sigma_a = 1.0
Q = sigma_a**2 * np.array([
    [dt**4/4, dt**3/2],
    [dt**3/2, dt**2]
])

# Ruido de medición
R = np.array([[sigma_z**2]])

# Estado inicial: posición inicial medida, velocidad 0
x = np.array([[z[0]],
              [0]])

# Covarianza inicial grande
P = np.eye(2) * 500

# Almacenar estimaciones
estimaciones = []

# ===========================================
# 4) Bucle del Filtro de Kalman
# ===========================================
for k in range(len(t)):

    # ---- Predicción ----
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    # ---- Corrección ----
    y_res = z[k] - (H @ x_pred)         # residuo
    S = H @ P_pred @ H.T + R            # cov residuo
    K = P_pred @ H.T @ np.linalg.inv(S) # ganancia de Kalman

    # Actualización del estado
    x = x_pred + K @ y_res
    P = (np.eye(2) - K @ H) @ P_pred

    estimaciones.append(x[0,0])

estimaciones = np.array(estimaciones)

# ===========================================
# 5) Gráficos finales
# ===========================================

plt.figure(figsize=(7,4))

# Mediciones ruidosas
plt.plot(t, z, color="blue", alpha=0.4, label="Mediciones", linewidth=1)

# Trayectoria real en AMARILLO
plt.plot(t, pos, color="yellow", linewidth=2, label="Real Track")

# Predicción Kalman Filter (rojo)
plt.plot(t, estimaciones, color="red", linewidth=2, label="Predicción Kalman Filter")

plt.title("Filtro de Kalman")
plt.xlabel("Tiempo (s)")
plt.ylabel("Posición (m)")
plt.legend()
plt.grid()
plt.show()

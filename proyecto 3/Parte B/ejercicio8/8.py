# ===========================================
# Filtro Complementario aplicado a la predicción Kalman
# ===========================================

import numpy as np
import matplotlib.pyplot as plt

# --- REGENERAMOS LOS DATOS REALES Y MEDICIONES RUIDOSAS ---
dt = 0.1
t = np.arange(0, 100, dt)
pos = 0.1*(3*t - t**2)   # trayectoria real

# Mediciones ruidosas
sigma_z = 20
z = pos + np.random.normal(0, sigma_z, len(pos))

# --- RECONSTRUIMOS EL KALMAN PARA USARLO AQUÍ ---
from numpy.linalg import inv

F = np.array([[1, dt],
              [0, 1]])

H = np.array([[1, 0]])

sigma_a = 1.0
Q = sigma_a**2 * np.array([
    [dt**4/4, dt**3/2],
    [dt**3/2, dt**2]
])

R = np.array([[sigma_z**2]])

x = np.array([[z[0]], [0]])
P = np.eye(2) * 500

estimaciones = []

for k in range(len(t)):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    y_res = z[k] - (H @ x_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ inv(S)

    x = x_pred + K @ y_res
    P = (np.eye(2) - K @ H) @ P_pred

    estimaciones.append(x[0,0])

estimaciones = np.array(estimaciones)

# ===========================================
# 1) APLICAR FILTRO COMPLEMENTARIO
# ===========================================

alpha = 0.95   

filtro_complementario = alpha * estimaciones + (1 - alpha) * z

# ===========================================
# 2) GRAFICAR RESULTADO
# ===========================================

plt.figure(figsize=(10,5))

plt.plot(t, z, alpha=0.3, label="Mediciones ruidosas", color="blue")
plt.plot(t, estimaciones, label="Filtro Kalman", color="red")
plt.plot(t, filtro_complementario, label=f"Filtro Complementario (alpha={alpha})", color="purple")
plt.plot(t, pos, label="Real Track", color="yellow", linewidth=2)

plt.xlabel("Tiempo (s)")
plt.ylabel("Posición (m)")
plt.grid()
plt.legend()
plt.title("Filtro Complementario aplicado al Filtro de Kalman")
plt.show()

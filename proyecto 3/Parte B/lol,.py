import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. Trayectoria real
# =============================
dt = 0.1
t = np.arange(0, 100, dt)
pos_real = 0.1 * (3*t - t**2)

# velocidad real (para simulación)
vel_real = np.gradient(pos_real, dt)

# =============================
# 2. Mediciones ruidosas
# =============================
np.random.seed(0)
ruido = np.random.normal(0, 20, size=len(pos_real))
pos_obs = pos_real + ruido

# =============================
# 3. Filtro de Kalman
# =============================

# Estado [pos, vel]
x = np.array([[0.0],
              [0.0]])

A = np.array([[1, dt],
              [0, 1]])

H = np.array([[1, 0]])

Q = np.array([[0.001, 0],
              [0, 0.001]])

R = np.array([[400]])

P = np.eye(2) * 500

pred_pos = []

for z in pos_obs:

    # ---- Predicción ----
    x = A @ x
    P = A @ P @ A.T + Q

    # ---- Actualización ----
    z = np.array([[z]])
    y = z - (H @ x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x = x + K @ y
    P = (np.eye(2) - K @ H) @ P

    pred_pos.append(x[0, 0])

pred_pos = np.array(pred_pos)

# =============================
# 4. Filtro Complementario
# =============================

alpha = 0.95   # AJUSTA ESTE VALOR para mejor resultado

comp_filter = alpha * pred_pos + (1 - alpha) * pos_obs

# =============================
# 5. Gráfica final
# =============================

plt.figure(figsize=(10, 5))
plt.title("Filtro de Kalman + Filtro Complementario", fontsize=14)

plt.plot(t, pos_obs, color='b', alpha=0.4, label="Mediciones (ruido)")
plt.plot(t, pos_real, color='r', linewidth=2, label="Real Track")
plt.plot(t, pred_pos, color='y', linewidth=2, label="Kalman Filter")
plt.plot(t, comp_filter, color='g', linewidth=2, label=f"Complementary Filter (alpha={alpha})")

plt.xlabel("Tiempo (s)")
plt.ylabel("Posición (m)")
plt.grid(True)
plt.legend()
plt.show()

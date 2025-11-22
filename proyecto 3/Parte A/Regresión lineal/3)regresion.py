import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Datos de Entrenamiento 
# X: Altura (variable independiente/predictora)
# Y: Peso (variable dependiente/a predecir)

# Datos de la tabla proporcionada
Altura_train = np.array([1.60, 1.65, 1.70, 1.73, 1.80])
Peso_train = np.array([60.0, 65.0, 72.3, 75.0, 80.0])

# El modelo de sklearn espera que X sea un array 2D. 
# Esto se logra con .reshape(-1, 1)
X_train = Altura_train.reshape(-1, 1)
Y_train = Peso_train

# Conjunto de prueba (Test Set) para predecir (Alturas proporcionadas en el enunciado)
Altura_test = np.array([1.58, 1.62, 1.69, 1.76, 1.82])
X_test = Altura_test.reshape(-1, 1)

# --- . Entrenamiento del Modelo de Regresión Lineal ---
# Se utiliza el módulo 'linear_model' y la clase 'LinearRegression' de sklearn
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, Y_train)

# --- . Predicción ---
# Predice el Peso para las alturas del conjunto de prueba (X_test)
Peso_predicho = modelo_regresion.predict(X_test)

# ---. Cálculo del RSS (Residual Sum of Squares) ---
# Para calcular el RSS, primero necesitamos predecir sobre los *mismos datos de entrenamiento*
# y luego calcular la suma de los errores al cuadrado (residuales).
Y_pred_train = modelo_regresion.predict(X_train)

# Cálculo del error cuadrático medio (MSE)
mse = mean_squared_error(Y_train, Y_pred_train)

# Cálculo del RSS
n = len(Y_train) # Número de observaciones en el conjunto de entrenamiento
RSS = mse * n


a = modelo_regresion.coef_[0]      # pendiente
b = modelo_regresion.intercept_    # intercepto

print("## Resultados del Modelo de Regresión Lineal ##")
print("-" * 50)
print(f"Pendiente (a): {a:.3f}")
print(f"Intercepto (b): {b:.3f}")

print(f"Ecuación del Modelo: Y = {a:.3f} * X + {b:.3f}")
print("-" * 50)

print("\n**Predicciones para el Conjunto de Prueba (Alturas):**")
for alt, peso in zip(Altura_test, Peso_predicho):
    print(f"Altura: {alt:.2f}m -> Peso Predicho: {peso:.2f}kg")

print("-" * 50)
print(f"**Suma de Cuadrados Residuales (RSS):** {RSS:.3f}")

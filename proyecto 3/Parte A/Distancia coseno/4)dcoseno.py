import numpy as np

def analizar_vectores(nombre_vector_1, vector_1, nombre_vector_2, vector_2):
    # Producto Punto (La suma de las multiplicaciones)
    producto_punto = np.dot(vector_1, vector_2)
    
    # Magnitudes del vector
    magnitud_vector_1 = np.linalg.norm(vector_1)
    magnitud_vector_2 = np.linalg.norm(vector_2)
    
    # Fórmula: (A . B) / (||A|| * ||B||)
    similitud_coseno = producto_punto / (magnitud_vector_1 * magnitud_vector_2)
    
    # Distancia coseno
    # Fórmula: 1 - Similitud
    distancia_coseno = 1 - similitud_coseno
    
    # Calculo de Ángulos
    # Definición: Similitud = cos(angulo)
    # Para despejar el 'angulo', aplicamos la función inversa del coseno (arcocoseno):
    # angulo = arccos(Similitud)
    #
    # IMPLEMENTACIÓN:
    # 1. np.clip(...): Asegura que el valor esté entre -1.0 y 1.0 (para evitar errores 
    #    numéricos si el resultado es 1.00000001).
    # 2. np.arccos(...): Nos da el ángulo en Radianes.
    # 3. np.degrees(...): Convierte esos Radianes a Grados . 
    angulo_en_radianes = np.arccos(np.clip(similitud_coseno, -1.0, 1.0))
    angulo_en_grados = np.degrees(angulo_en_radianes)
    
    print("Fórmula: Similitud coseno(A,B) = (A * B) / (||A|| * ||B||)")
    print("Fórmula: Distancia coseno(A,B) = 1 - Similitud Coseno")
    print(f"\n--- Análisis entre {nombre_vector_1} y {nombre_vector_2} ---")
    print(f"1. Producto Punto:     {producto_punto}")
    print(f"2. Magnitud de {nombre_vector_1}:   {magnitud_vector_1:.4f}")
    print(f"   Magnitud de {nombre_vector_2}:   {magnitud_vector_2:.4f}")
    print(f"3. Similitud Coseno:   {similitud_coseno:.4f}")
    print(f"4. Distancia Coseno:   {distancia_coseno:.4f}")
    print(f"5. Ángulo:             {angulo_en_grados:.2f}°")

# --- DEFINICIÓN DE TUS VECTORES ---

A = np.array([2, 1, 0, 2, 0, 1, 1, 1])
B = np.array([2, 1, 1, 1, 1, 0, 1, 1])

P = np.array([1, 2, 3, 0, 4, 6, 7, 9])
Q = np.array([2, 4, 3, 1, 8, 2, 4, 1])

S = np.array([2, 1, 4, 7, 1, 4, 5, 6])
T = np.array([3, 3, 3, 6, 1, 1, 7, 8])

# --- EJECUTAR EL ANÁLISIS ---
analizar_vectores("A", A, "B", B)
analizar_vectores("P", P, "Q", Q)
analizar_vectores("S", S, "T", T)

# --- EXPLICACIÓN TEÓRICA ---
print("\n=== EXPLICACIÓN TEÓRICA (Rúbrica Punto 4) ===")
print("Pregunta: ¿Qué significa si θ = 0 rad?")
print("Respuesta: Significa que los vectores están perfectamente alineados (son colineales).")
print("           Similitud = 1, Distancia = 0. Es la máxima similitud posible[cite: 11].")
print("-" * 30)
print("Pregunta: ¿Qué significa si θ = π/2 rad (90°)?")
print("Respuesta: Significa que los vectores son ortogonales (perpendiculares).")
print("           Similitud = 0, Distancia = 1. No comparten ninguna dirección en común[cite: 42].")
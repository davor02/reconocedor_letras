import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Leer secuencias por bloques exactos de 20 frames
def cargar_dinamicas(carpeta, frames_por_muestra=20):
    datos = []
    etiquetas = []

    for archivo in os.listdir(carpeta):
        if archivo.endswith(".csv"):
            ruta = os.path.join(carpeta, archivo)
            with open(ruta, "r") as f:
                reader = list(csv.reader(f))
                letra_actual = reader[0][0] if reader else None

                # Agrupar en bloques de 20 frames
                for i in range(0, len(reader) - frames_por_muestra + 1, frames_por_muestra):
                    bloque = reader[i:i + frames_por_muestra]
                    if len(bloque) == frames_por_muestra:
                        vector = []
                        for fila in bloque:
                            vector.extend([float(x) for x in fila[2:]])  # saltar letra y frame #
                        datos.append(vector)
                        etiquetas.append(letra_actual)
    return datos, etiquetas

# Cargar datos
X, y = cargar_dinamicas("datos_dinamicos")

print(f"✔ Muestras válidas cargadas: {len(X)}")

# Entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

score = modelo.score(X_test, y_test)
print(f"Precisión del modelo de letras dinámicas: {score * 100:.2f}%")

# Guardar modelo
with open("modelo_dinamicas.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("✅ Modelo de letras dinámicas guardado como 'modelo_dinamicas.pkl'")

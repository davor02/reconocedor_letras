
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Cargar letras estáticas
def cargar_estaticas(carpeta):
    datos = []
    etiquetas = []
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".csv"):
            ruta = os.path.join(carpeta, archivo)
            with open(ruta, "r") as f:
                reader = csv.reader(f)
                for fila in reader:
                    etiquetas.append(fila[0])
                    datos.append([float(x) for x in fila[1:]])
    return datos, etiquetas

# Cargar datos
X, y = cargar_estaticas("datos_señas")

# Entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

# Evaluación
score = modelo.score(X_test, y_test)
print(f"Precisión del modelo de letras estáticas: {score * 100:.2f}%")

# Guardar modelo
with open("modelo_estaticas.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("✅ Modelo de letras estáticas guardado como 'modelo_estaticas.pkl'")

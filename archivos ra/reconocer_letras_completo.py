import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque

# Cargar modelos
with open("modelo_estaticas.pkl", "rb") as f:
    modelo_estaticas = pickle.load(f)

with open("modelo_dinamicas.pkl", "rb") as f:
    modelo_dinamicas = pickle.load(f)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Configuración
buffer_frames = deque(maxlen=20)
umbral_movimiento = 0.03
ultima_letra = ""
consecutivos = 0

# Cámara
cap = cv2.VideoCapture(0)

def calcular_movimiento(p1, p2):
    if not p1 or not p2:
        return 0
    return sum(abs(a - b) for a, b in zip(p1, p2)) / len(p1)

print("📷 Reconociendo letras. Presiona ESC para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(rgb)

    letra_detectada = ""
    puntos_actuales = []

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            puntos_actuales = []
            for lm in hand_landmarks.landmark:
                puntos_actuales.extend([lm.x, lm.y, lm.z])
            buffer_frames.append(puntos_actuales)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if len(buffer_frames) == 20:
        mov = calcular_movimiento(buffer_frames[0], buffer_frames[-1])

        if mov > umbral_movimiento:
            secuencia = np.array(buffer_frames).flatten().reshape(1, -1)
            if secuencia.shape[1] == 1260:
                letra_detectada = modelo_dinamicas.predict(secuencia)[0]
        else:
            if puntos_actuales and len(puntos_actuales) == 63:
                letra_detectada = modelo_estaticas.predict([puntos_actuales])[0]

    if letra_detectada:
        if letra_detectada == ultima_letra:
            consecutivos += 1
        else:
            consecutivos = 0
        ultima_letra = letra_detectada

        if consecutivos > 3:
            cv2.putText(frame, f"Letra: {letra_detectada}", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow("Reconocimiento completo (A-Z)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

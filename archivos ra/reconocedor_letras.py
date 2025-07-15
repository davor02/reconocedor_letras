
import cv2
import mediapipe as mp
import numpy as np
import time

# Inicialización de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Variables para formar palabra
palabra = ""
ultima_letra = ""
tiempo_ultima = time.time()

def esta_en_zona(x, y):
    return 100 <= x <= 800 and 400 <= y <= 480

def detectar_letra(landmarks, w, h):
    # Simulación: Solo letra A como ejemplo
    dedo_indice = landmarks[8]
    dedo_medio = landmarks[12]
    pulgar = landmarks[4]

    if dedo_indice.y > pulgar.y and dedo_medio.y > pulgar.y:
        return "A"
    return ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    letra_actual = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener puntos clave
            landmarks = hand_landmarks.landmark
            dedo_indice = (int(landmarks[8].x * w), int(landmarks[8].y * h))

            # Detectar letra
            letra_actual = detectar_letra(landmarks, w, h)

            # Dibujar letra sobre el dedo
            if letra_actual:
                cv2.putText(frame, letra_actual, (dedo_indice[0], dedo_indice[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            # Verificar si la letra entra a la zona de escritura
            if letra_actual and esta_en_zona(dedo_indice[0], dedo_indice[1]):
                if letra_actual != ultima_letra or time.time() - tiempo_ultima > 1.5:
                    palabra += letra_actual
                    ultima_letra = letra_actual
                    tiempo_ultima = time.time()

    # Dibujar zona de escritura
    cv2.rectangle(frame, (100, 400), (800, 480), (255, 255, 255), 2)
    cv2.putText(frame, "Zona de escritura", (110, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Mostrar palabra
    cv2.putText(frame, f"Palabra: {palabra}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    # Mostrar cámara
    cv2.imshow("Lengua de Señas - RA", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

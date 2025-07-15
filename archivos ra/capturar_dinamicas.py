
import cv2
import mediapipe as mp
import csv
import os
import time

# Inicialización
letra_actual = input("Letra dinámica (ej: J, Z): ").upper()
os.makedirs("datos_dinamicos", exist_ok=True)
archivo_csv = f"datos_dinamicos/{letra_actual}_secuencias.csv"

# Parámetros
FRAMES_POR_MUESTRA = 20  # cantidad de frames que forman un movimiento
FPS = 10  # cuántos frames por segundo capturar

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Cámara
cap = cv2.VideoCapture(0)
print(f"Grabando secuencias para la letra dinámica '{letra_actual}'...")
print("Presiona 'r' para grabar una secuencia. ESC para salir.")

with open(archivo_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    muestra = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Mostrar
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Letra: {letra_actual} | Muestras: {muestra}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Captura de letras dinámicas", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        elif key == ord('r'):
            print(f"> Grabando secuencia de {FRAMES_POR_MUESTRA} frames...")
            secuencia = []

            for i in range(FRAMES_POR_MUESTRA):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    fila = [letra_actual, i]
                    for lm in hand.landmark:
                        fila.extend([lm.x, lm.y, lm.z])
                    secuencia.append(fila)
                time.sleep(1 / FPS)

            for fila in secuencia:
                writer.writerow(fila)

            muestra += 1
            print(f"✔ Secuencia {muestra} guardada")

cap.release()
cv2.destroyAllWindows()

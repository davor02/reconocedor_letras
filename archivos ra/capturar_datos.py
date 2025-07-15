
import cv2
import mediapipe as mp
import csv
import os

# Preguntar por la letra que se va a grabar
letra_actual = input("¿Qué letra estás grabando? (Ej: A): ").upper()

# Crear carpeta de salida si no existe
os.makedirs("datos_señas", exist_ok=True)
archivo_csv = f"datos_señas/{letra_actual}.csv"

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inicializar cámara
cap = cv2.VideoCapture(0)

contador = 0  # contador manual

# Abrir archivo para guardar landmarks
with open(archivo_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    print(f"Grabando datos para la letra '{letra_actual}'. Presiona 's' para guardar un ejemplo, ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostrar letra y cantidad de ejemplos guardados
        cv2.putText(frame, f"Letra: {letra_actual}  |  Guardados: {contador}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Mostrar cámara
        cv2.imshow("Captura de señas", frame)

        # Detectar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s') and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fila = [letra_actual]
                for lm in hand_landmarks.landmark:
                    fila.extend([lm.x, lm.y, lm.z])
                writer.writerow(fila)
                contador += 1
                print(f"[Ejemplo {contador} guardado]")

cap.release()
cv2.destroyAllWindows()

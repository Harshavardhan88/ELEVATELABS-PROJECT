# predict_live.py
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

IMG_SIZE = 96
CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
model = tf.keras.models.load_model('model/asl_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            x_min, y_min = max(0, x_min - 30), max(0, y_min - 30)
            x_max, y_max = min(w, x_max + 30), min(h, y_max + 30)

            roi = frame[y_min:y_max, x_min:x_max]
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue
            roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi_normalized = np.expand_dims(roi_resized / 255.0, axis=0)

            prediction = model.predict(roi_normalized, verbose=0)
            pred_letter = CLASSES[np.argmax(prediction)]

            cv2.putText(frame, pred_letter, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Real-Time ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

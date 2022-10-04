'''
Action Recognition with MediaPipe and LSTM Model
'''

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class ActionRecognition:
    def __init__(self, model_path, labels_path):
        self.model = load_model(model_path)
        self.labels = open(labels_path).read().strip().split('\n')
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                coords = []
                for data_point in hand_landmarks.landmark:
                    coords.append([data_point.x, data_point.y])
                coords = np.array(coords)
                coords = np.expand_dims(coords, axis=0)
                coords = np.expand_dims(coords, axis=3)
                prediction = self.model.predict(coords)
                prediction = prediction.argmax(axis=1)[0]
                label = self.labels[prediction]
                cv2.putText(
                    image, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
        return image


if __name__ == '__main__':
    model_path = 'model.h5'
    labels_path = 'labels.txt'
    action_recognition = ActionRecognition(model_path, labels_path)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = action_recognition.predict(frame)
        cv2.imshow('Action Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Your settings
plt.style.use('seaborn-paper')
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rc('font', size=18)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)
plt.rc('lines', markersize=16)
plt.rc('axes', grid=False)
warnings.filterwarnings('ignore')

# MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initializations
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)  # Bolder lines and larger points


def main():
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get results
        face_results = face_mesh.process(image_rgb)
        hand_results = hands.process(image_rgb)
        pose_results = pose.process(image_rgb)

        landmark_image = np.zeros_like(frame)

        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                mp_drawing.draw_landmarks(
                    image=landmark_image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                mp_drawing.draw_landmarks(
                    image=landmark_image,
                    landmark_list=hand_landmarks,
                    connections=mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=pose_results.pose_landmarks,
                connections=mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            mp_drawing.draw_landmarks(
                image=landmark_image,
                landmark_list=pose_results.pose_landmarks,
                connections=mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        cv2.imshow('Video Feed', frame)

        # Check for user input
        key = cv2.waitKey(1)
        if key == 27:  # Press ESC to exit
            break
        elif key == ord('s'):  # Press 's' to capture the current frame and landmarks
            # Plot side by side
            plt.figure(figsize=(10, 5))

            # Left plot (Original with landmarks)
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title('$\mathcal{I} = \{x_i\}_{i=1}^n$')
            plt.axis('off')

            # Right plot (Only landmarks)
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(landmark_image, cv2.COLOR_BGR2RGB))
            plt.title('$\mathcal{L} = \{x_i\}_{i=1}^n$')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig('comparison.png')
            plt.show()

    # Release the webcam and destroy windows
    face_mesh.close()
    hands.close()
    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

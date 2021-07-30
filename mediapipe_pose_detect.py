import cv2
import mediapipe 
import numpy as np

mp_drawing = mediapipe.solutions.drawing_utils
mp_holistic = mediapipe.solutions.holistic

cap = cv2.VideoCapture(0)

def main():

    holistic = mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence=0.5)

    while True:

        _, frame = cap.read()

        image_height, image_width, _ = frame.shape
        # Convert to RGB color space for model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        hand_frame = np.zeros((image_height, image_width, 3))

        # Draw Face
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

        # Draw Right Hand
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color = (0, 0, 255), thickness = 2, circle_radius = 3),
        mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 2, circle_radius = 4))

        # Draw Left Hand
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color = (255, 0, 0), thickness = 2, circle_radius = 3),
        mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 2, circle_radius = 4))

        # Draw Pose Detections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow("Frame", frame)
        cv2.imshow("Hand Movement", hand_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()

cap.release()
cv2.destroyAllWindows()
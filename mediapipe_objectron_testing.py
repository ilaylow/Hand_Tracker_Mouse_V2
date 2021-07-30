"""
This script utilises Google's Mediapipe Models for Objectron to test out on different sets of objects.
Purely Experimental. 
"""

import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

objectron = mp_objectron.Objectron(static_image_mode=False, max_num_objects = 5, min_detection_confidence = 0.5, min_tracking_confidence = 0.99, model_name = 'Cup')

camera = cv2.VideoCapture(0)

def main():
    while True:
        _, frame = camera.read()
        # Convert the BGR image to RGB.

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the frame as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = objectron.process(frame)

        # Draw the box landmarks on the frame.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(
                frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(frame, detected_object.rotation,
                                    detected_object.translation)

        cv2.imshow('MediaPipe Objectron', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
             break

if __name__ == "__main__":
    main()

camera.release()
cv2.destroyAllWindows()
"""
This script utilises Google's Mediapipe Models to perform a real time hand detection by detection appropriate
hand landmarks and uses these landmarks to map the movement to the computer's mouse, and enables a clicking
function by using a distance checker between the middle finger and thumb.

Scripts including use of mediapipe models has code derived from: https://google.github.io/mediapipe/solutions/solutions.html
"""

import cv2
import mediapipe 
import numpy as np
import pyautogui
import math

mp_drawing = mediapipe.solutions.drawing_utils
mp_holistic = mediapipe.solutions.holistic
mp_hands = mediapipe.solutions.hands

cap = cv2.VideoCapture(0)

def calc_dist(tup1, tup2):
    diff1 = (tup2[0] - tup1[0]) ** 2
    diff2 = (tup2[1] - tup1[1]) ** 2
    return math.sqrt(diff1 + diff2)

def main():

    holistic = mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence=0.5)

    while True:

        _, frame = cap.read()

        image_height, image_width, _ = frame.shape
        # Convert to RGB color space for model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        hand_frame = np.zeros((image_height, image_width, 3))


        if results.right_hand_landmarks is not None:
            right_hand_landmark_list = list(results.right_hand_landmarks.landmark)
            for landmark in right_hand_landmark_list:
                cv2.circle(frame, (round(landmark.x * image_width), round(landmark.y * image_height)), 3, (255, 0, 0), 2)
                cv2.circle(hand_frame, (round(landmark.x * image_width), round(landmark.y * image_height)), 3, (255, 255, 0), 3)
            
            index_finger_x_norm = right_hand_landmark_list[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            index_finger_y_norm = right_hand_landmark_list[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_finger_x_norm = right_hand_landmark_list[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
            middle_finger_y_norm = right_hand_landmark_list[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            thumb_tip_x_norm = right_hand_landmark_list[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_tip_y_norm = right_hand_landmark_list[mp_hands.HandLandmark.THUMB_TIP].y

            index_finger_x = round(index_finger_x_norm * image_width)
            index_finger_y = round(index_finger_y_norm * image_height)

            middle_finger_x = round(middle_finger_x_norm * image_width)
            middle_finger_y = round(middle_finger_y_norm * image_height)

            thumb_tip_x = round(thumb_tip_x_norm * image_width)
            thumb_tip_y = round(thumb_tip_y_norm * image_height)

            cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 4, (128,0,128), 3)
            cv2.circle(frame, (index_finger_x, index_finger_y), 4, (0, 255, 0), 3)
            cv2.circle(frame, (middle_finger_x, middle_finger_y), 4, (0, 255, 255), 3)



            # Check distance between thumb tip and index finger
            thumb_middle_dist = calc_dist((middle_finger_x_norm, middle_finger_y_norm), (thumb_tip_x_norm, thumb_tip_y_norm))
            if thumb_middle_dist < 0.05:
                cv2.putText(frame, "Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(hand_frame, "Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                x, y = pyautogui.position()
                #pyautogui.leftClick(x, y)
                print("Clickity Clack Clack Clack")
            
            pyautogui.moveTo(index_finger_x * 3, index_finger_y * 2.25)


        cv2.imshow("Frame", frame)
        cv2.imshow("Hand Movement", hand_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()

cap.release()
cv2.destroyAllWindows()
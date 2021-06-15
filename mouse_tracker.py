import cv2
import os
import numpy as np
import pickle
import math
import pyautogui

# Define ROI Regions
roi_lower_X = 15
roi_upper_X = 300

roi_lower_Y = 90
roi_upper_Y = 380

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

#Load in model histogram
model_file = open("model_hist.pkl", 'rb')
model_hist = pickle.load(model_file)
model_file.close()

def convolve(B, r):
    D = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
    cv2.filter2D(B, -1, D, B)
    return B

while True:
    ret, frame = cap.read()

    # Define ROI
    roi = frame[roi_lower_Y: roi_upper_Y, roi_lower_X: roi_upper_X]
    
    # Params are (img, lower right coord, upper left coord, rgb color, thickness)
    cv2.rectangle(frame, (roi_lower_X, roi_lower_Y), (roi_upper_X, roi_upper_Y), (0, 255, 0), 5)
    cv2.putText(frame, 'Region Of Interest', (10, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Testing out backprojection
    new_img = cv2.calcBackProject([hsv_roi], channels=[0,1], hist= model_hist, ranges=[0,180,0,256], scale=1)
    new_img = convolve(new_img, r = 5)
    
    # Perform image erosion
    kernel = np.ones((3, 3), np.uint8)
    new_img = cv2.erode(new_img, kernel)
    blur_mask = cv2.blur(new_img, (3, 3))
    _, new_img_thresh = cv2.threshold(blur_mask, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(new_img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        max_contour = max(contours, key = cv2.contourArea)
        return_hull = cv2.convexHull(max_contour)
        return_hull_indices = cv2.convexHull(max_contour, returnPoints = False)

        for point in return_hull:
            point = point[0]
            cv2.circle(roi, tuple(point), 1, [0, 0, 255], 3)
            #print(point)

        dist_hull = [(math.dist(return_hull[i][0], return_hull[i + 1][0]), (return_hull[i][0], return_hull[i + 1][0])) for i in range(len(return_hull) - 1)]
        dist_hull.sort(reverse = True, key = lambda x: x[0])

        # Get the top two distances
        top_dist_2 = dist_hull[:2]
        for _, points in top_dist_2:
            p1, p2 = points
            cv2.line(roi, tuple(p1), tuple(p2), [0, 255, 0], 2)

        print(top_dist_2)

        p1 = top_dist_2[0][1][1]
        p2 = top_dist_2[1][1][0]

        # Get midpoint of p1 and p2
        midpoint_mouse = (round((p1[0] + p2[0]) / 2), round((p1[1] + p2[1]) / 2))
        cv2.circle(roi, midpoint_mouse, 1, [106, 13, 173], 3)
        dx = midpoint_mouse[0]
        dy = midpoint_mouse[1]

        # Find a way to stabilise the mouse due to fluctuations 
        pyautogui.moveTo(dx * 6.736, dy * 5.684)

    cv2.imshow("roi", roi)
    cv2.imshow("frame", frame)
    cv2.imshow("BackProj Mask", new_img_thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
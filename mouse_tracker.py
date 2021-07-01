import cv2
import os
import numpy as np
import pickle
import math
import pyautogui

from BackgroundSubtract import BackGroundSubtract
from BackgroundSubtract import BLUR_RADIUS, roi_lower_X, roi_lower_Y, roi_upper_X, roi_upper_Y, erode_kernel, dilate_kernel

# TODO: Use Both Background Subtraction and BackProjection and use bitwise and function to create a proper mask
# TODO: Find new way to calculate point and stabilise it

pyautogui.FAILSAFE = False

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

def calculate_centroid(img_thresh):
    moments = cv2.moments(img_thresh)

    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return int(cX), int(cY)

    return int((roi_lower_X + roi_upper_X) / 2), int((roi_lower_Y + roi_upper_Y) / 2)

gray_background_roi = BackGroundSubtract.read_initial_background(cap)
print("Finished Reading Background...Enter hand in front of camera now!")

while True:
    ret, frame = cap.read()

    # Define ROI
    roi = frame[roi_lower_Y: roi_upper_Y, roi_lower_X: roi_upper_X]

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    """ # Testing out backprojection
    new_img = cv2.calcBackProject([hsv_roi], channels=[0,1], hist= model_hist, ranges=[0,180,0,256], scale=1)
    new_img = convolve(new_img, r = 5)
    
    # Perform image erosion
    new_img = cv2.erode(new_img, erode_kernel, iterations = 1)
    blur_mask = cv2.blur(new_img, (3, 3))
    _, backproj_img_thresh = cv2.threshold(blur_mask, 0, 255, cv2.THRESH_BINARY) """

    gray_roi = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (BLUR_RADIUS, BLUR_RADIUS), 0)

    diff, subtract_img_thresh = BackGroundSubtract.perform_background_subtraction(gray_roi, gray_background_roi, use_external=False)

    cX, cY = calculate_centroid(subtract_img_thresh)
    
    # Draw circle around centroid calculated
    cv2.circle(roi, (cX, cY), 2, (255, 0, 0), 3)

    # We can bitwise_and the subtract_img_thresh and backprog_img_thresh
    #combined_img_thresh = cv2.bitwise_and(backproj_img_thresh, subtract_img_thresh, mask = None)

    contours, _ = cv2.findContours(subtract_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        max_contour = max(contours, key = cv2.contourArea)
        return_hull = cv2.convexHull(max_contour)
        return_hull_indices = cv2.convexHull(max_contour, returnPoints = False)

        maxDist_Point = (0, (0, 0))
        for point in return_hull:
            point = point[0]
            cv2.circle(roi, tuple(point), 1, [0, 0, 255], 3)
            dist_from_centroid = math.dist(tuple(point), (cX, cY))
            newPoint = (dist_from_centroid, tuple(point))
            maxDist_Point = max(newPoint, maxDist_Point ,key=lambda x:x[0])

        print(maxDist_Point)

        # Draw line from centroid to furthest convex hull point
        mouse_point = maxDist_Point[1]
        cv2.line(roi, (cX, cY), mouse_point, [0, 255, 0], 2)
        cv2.circle(roi, mouse_point, 2, (255, 255, 0), 3)
            #print(point)

        

        """ dist_hull = [(math.dist(return_hull[i][0], return_hull[i + 1][0]), (return_hull[i][0], return_hull[i + 1][0])) for i in range(len(return_hull) - 1)]
        dist_hull.sort(reverse = True, key = lambda x: x[0])

        # Get the top two distances
        top_dist_2 = dist_hull[:2]
        for _, points in top_dist_2:
            p1, p2 = points
            cv2.line(roi, tuple(p1), tuple(p2), [0, 255, 0], 2)

        p1 = top_dist_2[0][1][1]
        p2 = top_dist_2[1][1][0]

        # Get midpoint of p1 and p2
        midpoint_mouse = (round((p1[0] + p2[0]) / 2), round((p1[1] + p2[1]) / 2))
        cv2.circle(roi, midpoint_mouse, 1, [106, 13, 173], 3)
        dx = midpoint_mouse[0]
        dy = midpoint_mouse[1] """

        # Find a way to stabilise the mouse due to fluctuations 
        pyautogui.moveTo(mouse_point[0] * 6.736, mouse_point[1] * 5.684)
    
    # Params are (img, lower right coord, upper left coord, rgb color, thickness)
    cv2.rectangle(frame, (roi_lower_X, roi_lower_Y), (roi_upper_X, roi_upper_Y), (0, 255, 0), 3)
    cv2.putText(frame, 'Region Of Interest', (10, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("roi", roi)
    cv2.imshow("frame", frame)
    cv2.imshow("Subtract Hand Mask", subtract_img_thresh)
    #cv2.imshow("BackProj Hand Mask", backproj_img_thresh)
    #cv2.imshow("Combined Img Thresh", combined_img_thresh)
    cv2.imshow("Diff", diff)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif cv2.waitKey(1) & 0xFF == ord('r'):
        gray_background_roi = BackGroundSubtract.read_initial_background(cap)
        print("Retook Background Frames...")

cv2.destroyAllWindows()
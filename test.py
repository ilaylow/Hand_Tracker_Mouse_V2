import cv2
import pickle
import numpy as np
import os

# Argument represents camera number
cap = cv2.VideoCapture(0)

# Define ROI Regions
roi_lower_X = 15
roi_upper_X = 300

roi_lower_Y = 90
roi_upper_Y = 380

max_hsv = [160, 198, 255]
min_hsv = [60, 87, 119]

font = cv2.FONT_HERSHEY_SIMPLEX

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

    #Load in model histogram
    model_file = open("model_hist.pkl", 'rb')
    model_hist = pickle.load(model_file)
    model_file.close()

    # Testing out backprojection
    new_img = cv2.calcBackProject([hsv_roi], channels=[0,1], hist= model_hist, ranges=[0,180,0,256], scale=1)
    new_img = convolve(new_img, r = 5)
    
    # Perform image erosion
    kernel = np.ones((2, 2), np.uint8)
    new_img = cv2.erode(new_img, kernel)
    blur_mask = cv2.blur(new_img, (2, 2))
    _, new_img_thresh = cv2.threshold(blur_mask, 0, 255, cv2.THRESH_BINARY)

    contours, val1 = cv2.findContours(new_img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        continue

    max_contour = max(contours, key = cv2.contourArea)
    return_hull = cv2.convexHull(max_contour, returnPoints = False)
    
    hulls = [cv2.convexHull(contour) for contour in contours]
    hull_indices = [cv2.convexHull(contour, returnPoints = False) for contour in contours]

    max_hull = max(hulls, key = cv2.contourArea)
    try:   
        defects = cv2.convexityDefects(max_contour, return_hull)
    except:
        defects = None
        print("Stupid Monotonous Points aren't showing due to self intersections. FUCK")

    cv2.drawContours(roi, max_contour, -1, (0, 255, 0))
    cv2.drawContours(roi, [max_hull], -1, (255, 0, 0))

    # Code adapted from: https://theailearner.com/2020/11/09/convexity-defects-opencv/
    if defects is not None:
        for i in range(defects.shape[0]):
            _, _, f, _ = defects[i, 0]
            defect_point = tuple(max_contour[f][0])
            cv2.circle(roi, defect_point, 5, [0, 0, 255], -1)
        
        # Use these convexDefects to determine the number of fingers
        print(defects.shape)

    cv2.imshow("roi", roi)
    cv2.imshow("frame", frame)
    cv2.imshow("BackProj Mask", new_img_thresh)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
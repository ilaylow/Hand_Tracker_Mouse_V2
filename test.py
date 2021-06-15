import cv2
import pickle
import numpy as np
import math
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

LINE_THRESH = 80

font = cv2.FONT_HERSHEY_SIMPLEX

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
    number_shown = 0

    # Define ROI
    roi = frame[roi_lower_Y: roi_upper_Y, roi_lower_X: roi_upper_X]

    # Params are (img, lower right coord, upper left coord, rgb color, thickness)
    cv2.rectangle(frame, (roi_lower_X, roi_lower_Y), (roi_upper_X, roi_upper_Y), (0, 255, 0), 5)
    cv2.putText(frame, 'Region Of Interest', (10, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Testing out backprojection
    new_img = cv2.calcBackProject([hsv_roi], channels=[0,1], hist= model_hist, ranges=[0,180,0,256], scale=1)
    new_img = convolve(new_img, r = 3)
    
    # Perform image erosion
    kernel = np.ones((3, 3), np.uint8)
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
            #cv2.circle(roi, defect_point, 5, [0, 0, 255], -1)
        
        # Use these convexDefects to determine the number of fingers
        # Make the assumption that the consecutive contours with the biggest distance between them will
        # be the ones with fingers
        defects_point_dist = []
        for i in range(defects.shape[0]):
            start, end, far, _ = defects[i, 0]
            line_start = tuple(max_contour[start][0])
            line_far = tuple(max_contour[far][0])
            line_end = tuple(max_contour[end][0])
            cv2.line(roi,line_start, line_far,[106, 13, 173],2)
            cv2.line(roi, line_far, line_end, [106, 13, 173], 2)

            # Get a list of 2D Tuples, one thats the euclid dist, and the other the set of point
            # The Euclid Dist is the average of the two adjacent lines euclid (Doesn't work as well)
            """ euclid_dist_avg = (math.dist(line_start, line_far) + math.dist(line_far, line_end)) / 2
            defects_point_dist.append(euclid_dist_avg) """

            # Using cosine rule to determine angle between fingers so we can use it to determine number of fingers
            # Angle = cos^-1(a^2 + b^2 - c^2 / 2ab)
            
            #Calculate line lengths
            a = math.dist(line_far, line_start)
            b = math.dist(line_end, line_far)
            c = math.dist(line_start, line_end)

            angle = math.acos((pow(a, 2) + pow(b, 2) - pow(c, 2)) / (2*a*b))
            
            # Check distance for C

            ## Need to find more robust method for detecting the true convexity defects on hand
            # Could try some distance normalisation techniques (doesn't actually work very well)
            # Looking at relative distances could be an option
            norm_a = a / math.sqrt(roi.shape[0] ** 2 + roi.shape[1] ** 2)
            norm_b = b / math.sqrt(roi.shape[0] ** 2 + roi.shape[1] ** 2)
            norm_c = c / math.sqrt(roi.shape[0] ** 2 + roi.shape[1] ** 2)
            if angle < math.pi / 2 and norm_c >= 0.10:
                print(f"A: {norm_a}")
                print(f"B: {norm_b}")
                print(f"C: {norm_c}")
                cv2.circle(roi, (line_far), 5, [0, 0, 255], -1)
                number_shown+=1
                

        max_dist_hull = max([math.dist(max_hull[i][0], max_hull[i+1][0]) for i in range(len(max_hull) - 1)])
        if max_dist_hull >= 100:
            number_shown += 1

    cv2.putText(frame, 'Number: ' + str(number_shown), (20, 130), font, 1, (200, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("roi", roi)
    cv2.imshow("frame", frame)
    cv2.imshow("BackProj Mask", new_img_thresh)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
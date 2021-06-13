import cv2
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

while True:
    ret, frame = cap.read()

    # Define ROI
    roi = frame[roi_lower_Y: roi_upper_Y, roi_lower_X: roi_upper_X]

    # Params are (img, lower right coord, upper left coord, rgb color, thickness)
    cv2.rectangle(frame, (roi_lower_X, roi_lower_Y), (roi_upper_X, roi_upper_Y), (0, 255, 0), 5)
    cv2.putText(frame, 'Region Of Interest', (10, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv_roi, tuple(min_hsv), tuple(max_hsv))

    # Testing out blurring and thresholding to remove noise
    blur_mask = cv2.blur(skin_mask, (2, 2))
    ret, thresh_mask = cv2.threshold(blur_mask, 0, 255, cv2.THRESH_BINARY)

    contours, val1 = cv2.findContours(thresh_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        continue

    max_contour = max(contours, key = cv2.contourArea)
    return_hull = cv2.convexHull(max_contour)
    hull = [cv2.convexHull(contour) for contour in contours]

    for i in range(len(contours)):
        cv2.drawContours(roi, contours, i, (0, 255, 0))
        cv2.drawContours(roi, hull, i, (0, 255, 0))


    cv2.imshow("roi", roi)
    cv2.imshow("frame", frame)
    cv2.imshow("skin mask", skin_mask)
    cv2.imshow("Blurred & Thresholded Mask", thresh_mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
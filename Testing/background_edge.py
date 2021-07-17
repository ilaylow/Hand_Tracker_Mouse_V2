import cv2
import numpy as np
import random

from numpy.core.defchararray import upper

roi_lower_X = 15
roi_upper_X = 300

roi_lower_Y = 90
roi_upper_Y = 380

blur = 21
canny_low = 15
canny_high = 150
min_area = 0.0005
max_area = 0.95
dilate_iter = 10
erode_iter = 10
mask_color = (0.0,0.0,0.0)

cap = cv2.VideoCapture(0)

# Perform foreground detection with hand
def main():

    while True:

        ret, frame = cap.read()
        
        roi = frame[roi_lower_Y: roi_upper_Y, roi_lower_X: roi_upper_X]  
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Using Canny Edge Detection
        edges = cv2.Canny(gray_roi, canny_low, canny_high)

        # Apply operations
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if contours is not None:
            contour_area = [(c, cv2.contourArea(c)) for c in contours]
            
            # Get area of image?
            image_area = roi.shape[0] * roi.shape[1]

            upper_bound_area = max_area * image_area
            lower_bound_area = min_area * image_area

            mask = np.zeros(edges.shape, dtype = np.uint8)

            for contour in contour_area:
                if contour[1] > lower_bound_area and contour[1] < upper_bound_area:
                    mask = cv2.fillConvexPoly(mask, contour[0], (255))

            # Dilate and erode and blur to smooth mask
            mask = cv2.dilate(mask, None, iterations = dilate_iter)
            mask = cv2.erode(mask, None, iterations = erode_iter)
            mask = cv2.GaussianBlur(mask, (blur, blur), 0)

            mask_stack = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_stack = mask_stack.astype('float32') / 255.0
            roi = roi.astype('float32') / 255.0

            masked = (mask_stack * roi) + ((1-mask_stack) * mask_color)
            masked = (masked * 255).astype('uint8')
            

        cv2.rectangle(frame, (roi_lower_X, roi_lower_Y), (roi_upper_X, roi_upper_Y), (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        cv2.imshow("Edges", edges)
        cv2.imshow("Foreground", masked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return

if __name__ == "__main__":
    main()

cv2.destroyAllWindows()
cap.release()
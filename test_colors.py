import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
OFFSET = 3

# Convert image to YCrCb, Cr and Cb are constructed from subtracting luma from RGB red and blue components
# The skin pixels form a compact cluster around Cb-Cr Plane
# Adapted from: https://nalinc.github.io/blog/2018/skin-detection-python-opencv/

while True:
    ret, frame = cap.read()

    roi_lower_X = 100
    roi_upper_X = 150

    roi_lower_Y = 100
    roi_upper_Y = 280

    roi = frame[roi_lower_Y + OFFSET: roi_upper_Y - OFFSET, roi_lower_X + OFFSET: roi_upper_X - OFFSET]
    cv2.rectangle(frame, (roi_lower_X, roi_lower_Y), (roi_upper_X, roi_upper_Y), (0, 255, 0), 1)
    cv2.putText(frame, 'Press T To Take Photo & Generate Histogram', (10, 70), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    cv2.imshow("roi", roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('t'):
        
        # Generate Histogram and Save to Pickle File
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        histogram = cv2.calcHist([roi_hsv], channels=[0, 1], mask=None, histSize = [80, 256], ranges = [0, 180, 0, 256])
        print(type(histogram))

        colors = ("r", "g", "b")
        for i, color in enumerate(colors):
            hist = cv2.calcHist([roi_hsv],[i],None,[256],[0,256])
            plt.plot(hist ,color = color)
            plt.xlim([0,256])
            plt.ylim([0,500])
            
        plt.savefig(r"model_hist_plot.png")
        
        file_obj = open("model_hist.pkl", 'wb')
        pickle.dump(histogram, file_obj)
        file_obj.close()

        break

cv2.destroyAllWindows()


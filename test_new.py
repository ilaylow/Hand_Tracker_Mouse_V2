import cv2
import os
import pickle

def convolve(B, r):
    D = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
    cv2.filter2D(B, -1, D, B)
    return B

test_img = cv2.imread("hand1.jpg", cv2.IMREAD_COLOR)
hsv_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)

#Load in model histogram
model_file = open("model_hist.pkl", 'rb')
model_hist = pickle.load(model_file)
model_file.close()

new_img = cv2.calcBackProject([hsv_img], channels=[0,1], hist= model_hist, ranges=[0,180,0,256], scale=1)

new_img = convolve(new_img, r = 3)
_, new_img_thresh = cv2.threshold(new_img, 10, 255, cv2.THRESH_BINARY)

cv2.imshow("hand_image", new_img_thresh)
cv2.waitKey(0)

cv2.destroyAllWindows()
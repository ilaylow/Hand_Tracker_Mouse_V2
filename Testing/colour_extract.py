import cv2
import numpy as np

def nothing(x):
    pass


#Creates a trackbar to adjust HSV values
cv2.namedWindow("Range Modifier")

cv2.createTrackbar("Lower - H", "Range Modifier", 0, 179, nothing)
cv2.createTrackbar("Lower - S", "Range Modifier", 0, 255, nothing)
cv2.createTrackbar("Lower - V", "Range Modifier", 0, 255, nothing)
cv2.createTrackbar("High - H", "Range Modifier", 179, 179, nothing)
cv2.createTrackbar("High - S", "Range Modifier", 255, 255, nothing)
cv2.createTrackbar("High - V", "Range Modifier", 255, 255, nothing)

lh = 33
ls = 0
lv = 70
hh = 73
hs = 255
hv = 255

cap = cv2.VideoCapture(0)
#heatmap = cv2.applyColorMap(image, cv2.COLORMAP_HSV)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#heatmap = cv2.resize(heatmap, (700, 700))
#gray = cv2.resize(gray, (700, 700))

while True:

    ret, frame = cap.read()

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the values set from the trackbar
    lh = cv2.getTrackbarPos("Lower - H", "Range Modifier")
    ls = cv2.getTrackbarPos("Lower - S", "Range Modifier")
    lv = cv2.getTrackbarPos("Lower - V", "Range Modifier")
    hh = cv2.getTrackbarPos("High - H", "Range Modifier")
    hs = cv2.getTrackbarPos("High - S", "Range Modifier")
    hv = cv2.getTrackbarPos("High - V", "Range Modifier")

    lower_green = np.array([lh, ls, lv])
    higher_green = np.array([hh, hs, hv])

    mask = cv2.inRange(hsv_img, lower_green, higher_green)

    cv2.imshow('mask', cv2.resize(mask, (500, 500)))
    cv2.imshow('original', cv2.resize(frame, (500, 500)))
    #cv2.imshow('heatmap', heatmap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

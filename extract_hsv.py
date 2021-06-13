import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)

RESET_FRAME = 100
frame_count = 0

RANGE_HSV = 50

# Draw a row and column of rectangles to obtain hsv values
NUM_RECTS = 5

rect_gap = 10
col_gap = 15
rect_len = 3
start_coord = [100, 200]
font = cv2.FONT_HERSHEY_SIMPLEX

hsv_values_rects = []
max_hsv = [0, 0, 0]
min_hsv = [255, 255, 255]

def extract_average_hsv_values(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    count = 0

    avg_h = 0
    avg_s = 0
    avg_v = 0
    for i in range(len(h)):
        pixel_h = h[i]
        pixel_s = s[i]
        pixel_v = v[i]
        for j in range(len(pixel_h)):
            avg_h += pixel_h[j]
            avg_s += pixel_s[j]
            avg_v += pixel_v[j]
            count += 1
    
    avg_h = round(avg_h / count)
    avg_s = round(avg_s / count)
    avg_v = round(avg_v / count)

    hsv_values_rects.append([avg_h, avg_s, avg_v])

    return [avg_h, avg_s, avg_v]


while True:
    ret, frame = cap.read()

    for i in range(NUM_RECTS):
        rect_coord = [start_coord[0], start_coord[1] + (i * rect_gap)]
        for j in range(NUM_RECTS):
            rect_coord = (rect_coord[0] + col_gap, rect_coord[1])
            extract_average_hsv_values(frame[rect_coord[0]: rect_coord[0] + rect_len, rect_coord[1]: rect_coord[1] + rect_len])
            cv2.rectangle(frame, rect_coord, (rect_coord[0] + rect_len, rect_coord[1] + rect_len), (0, 255, 0), 1)
    
    

    # Get max range of hsv values
    """ for arr in hsv_values_rects:
        arr_sum = sum(arr)
        if (arr_sum > sum(max_hsv)):
            max_hsv = arr
        if (arr_sum < sum(min_hsv)):
            min_hsv = arr """
    
    avg_h = round(sum([arr[0] for arr in hsv_values_rects]) / len(hsv_values_rects))
    avg_s = round(sum([arr[1] for arr in hsv_values_rects]) / len(hsv_values_rects))
    avg_v = round(sum([arr[2] for arr in hsv_values_rects]) / len(hsv_values_rects))

    test_minh = max(min(avg_h - RANGE_HSV, min_hsv[0]), 0)
    test_mins = max(min(avg_s - RANGE_HSV, min_hsv[1]), 0)
    test_minv = max(min(avg_v - RANGE_HSV, min_hsv[2]), 0)

    test_maxh = max(avg_h + RANGE_HSV, max_hsv[0])
    test_maxs = max(avg_s + RANGE_HSV, max_hsv[1])
    test_maxv = max(avg_v + RANGE_HSV, max_hsv[2])

    max_hsv = [test_maxh, test_maxs, test_maxv]
    min_hsv = [test_minh, test_mins, test_minv]

    #print(f"Max HSV value: {max_hsv}")
    #print(f"Min HSV value: {min_hsv}")

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv_frame, tuple(min_hsv), tuple(max_hsv))

    blurred_skin = cv2.blur(skin_mask, (2, 2))
    ret, threshold_skin = cv2.threshold(blurred_skin, 0, 255, cv2.THRESH_BINARY)

    if frame_count == RESET_FRAME:
        print("AWH YEETUS RESETTING")
        print(f"Max HSV value: {max_hsv}")
        print(f"Min HSV value: {min_hsv}")
        min_hsv = [255, 255, 255]
        max_hsv = [0, 0, 0]
        hsv_values_rects = []
        frame_count = 0
    
    frame_count += 1

    #cv2.imshow("thresh mask", threshold_skin)
    cv2.imshow("skin mask", skin_mask)
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
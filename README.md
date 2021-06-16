# Hand_Tracker_Mouse_V2

**Note that the above is still in its experimental changes as the algorithm lacks stability.**

The code above is an attempt to experiment with skin detection with different colorspaces through Image BackProjection with the OpenCV library for Python. 

This is transformed into two applications, **finger counting** and **mouse tracking** with OpenCV. There will be instructions provided below on how to use these algorithms for your own experimental purposes.

## Package Installation

Ensure that you have the latest versions of the following packages:
* NumPy
* OpenCV
* Pickle
* Pyautogui

To install these, perform the following:
```bash
pip install numpy
pip install opencv-python
pip install pickle
pip install pyautogui
```

## Usage

As skin and lighting may differ depending on usage, the code utilises backprojection to tackle this variance. More information can be found here: https://docs.opencv.org/master/dc/df6/tutorial_py_histogram_backprojection.html

First run test_colors.py and attempt to fit hand within rectangle, as shown in the image below:
![Extract Histogram For BackProjection](/images/showing_extract_histogram.png)

Upon pressing, two new files should be generated: **model_hist_plot.png** and **model_hist.pkl**. The model_hist_plot image shows the color distribution of the HSV colorspace (Hue, Saturation, Value).

Next we can either run, count_fingers.py or mouse_tracker.py. The image below demonstrates the use of count_fingers.py.

![Using Count_Fingers.py](/images/counting_fingers.png)

Lastly, please note that performance may differ heavily depending on background environment and light. Any new suggestions for other algorithmic implementations are always welcome. Improvements are being worked on.

**Have fun!**




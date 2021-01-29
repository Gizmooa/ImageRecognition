import cv2 as cv
import numpy as np
import os
from time import time
from wincap import WinCap
from vision import Vision
from hsvfilter import HsvFilter

# HSV thresholding
# Image processing technique 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize the WindowCapture class
wincap = WinCap('BlueStacks')

oreVision = Vision('hsvNeedle.jpg')
oreVision.init_control_gui()

hsvFilter = HsvFilter(13,131,45,15,185,255,40,0,0,0)

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    # image pre-processing
    # Use without filter to play around with hsv filter settings
    #processedImage = oreVision.apply_hsv_filter(screenshot)
    processedImage = oreVision.apply_hsv_filter(screenshot, hsvFilter)

    # Detects locations of objects on processed image
    rectList = oreVision.findObject(processedImage, 0.35)

    # Draw rectangles onto the detected objects
    output_img = oreVision.displayRectangles(screenshot, rectList)

    # Display processed image
    cv.imshow('Matches', output_img)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
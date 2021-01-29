import cv2 as cv
import numpy as np
import os
from time import time
from wincap import WinCap
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter

# Image processing with edge filter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize the WindowCapture class
wincap = WinCap('BlueStacks')

oreVision = Vision('edgeNeedle.jpg')
oreVision.init_control_gui()

hsvFilter = HsvFilter(13,131,45,15,185,255,40,0,0,0)

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    # image pre-processing
    # Use without filter to play around with hsv filter settings
    #processedImage = oreVision.apply_hsv_filter(screenshot)
    processedImage = oreVision.apply_edge_filter(screenshot)

    # Detects locations of objects on processed image
    #rectList = oreVision.findObject(processedImage, 0.35)

    # Draw rectangles onto the detected objects
    #output_img = oreVision.displayRectangles(screenshot, rectList)

    # keypoint searching
    keypoint_image = processedImage
    # crop the image to remove the ui elements
    x, w, y, h = [200, 1130, 70, 750]
    keypoint_image = keypoint_image[y:y+h, x:x+w]

    kp1, kp2, matches, match_points = oreVision.match_keypoints(keypoint_image)
    match_image = cv.drawMatches(
        oreVision.needle, 
        kp1, 
        keypoint_image, 
        kp2, 
        matches, 
        None)

    if match_points:
        # find the center point of all the matched features
        center_point = oreVision.centeroid(match_points)
        # account for the width of the needle image that appears on the left
        center_point[0] += oreVision.needleW
        # drawn the found center point on the output image
        match_image = oreVision.displayCrossHair(match_image, [center_point])

    # Display processed image
    cv.imshow('Keypoint Search', match_image)
    #cv.imshow('Processed', processedImage)
    #cv.imshow('Matches', processedImage)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
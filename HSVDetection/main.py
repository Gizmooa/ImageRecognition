import cv2 as cv
import numpy as np
import os
from time import time
from wincap import WinCap
from vision import Vision
from hsvfilter import HsvFilter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Trigger if you dont want to see the vision
SHOW_VISION = True

# vision object containing the needle image
oreVision = Vision('hsvNeedle.jpg')

# hsv filter filtering copper ores
hsvFilter = HsvFilter(13,131,45,15,185,255,40,0,0,0)

# initiate the windows capture class
wincap = WinCap('BlueStacks', hsvFilter, oreVision)

# start a thread updating the screenshot and processed screenshot
wincap.start()

loop_time = time()
print("Started session...")
while(True):

    # Only start finding objects when there exists a screenshot
    if wincap.screenshot is None:
        continue

    # Detects locations of objects on processed image
    rectList = oreVision.findObject(wincap.processedScreenshot, 0.4)

    # Draw rectangles onto the detected objects
    output_img = oreVision.displayRectangles(wincap.screenshot, rectList)

    if (SHOW_VISION):
        cv.imshow('Vision', output_img)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # Press q to stop the detection
    if cv.waitKey(1) == ord('q'):
        wincap.stop()
        cv.destroyAllWindows()
        break

print('Stopped session...')
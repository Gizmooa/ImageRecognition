import cv2 as cv
import numpy as np
import os
from time import time
from wincap import WinCap
from vision import Vision

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize the WindowCapture class
wincap = WinCap('BlueStacks')

oreVision = Vision('needle.jpg')

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    processedImage = oreVision.findObject(screenshot, 0.5, 'rectangles')

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
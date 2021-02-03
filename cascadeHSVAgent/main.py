import cv2 as cv
import numpy as np
import os
import pyautogui
from time import time, sleep
from wincap import WinCap
from vision import Vision
from hsvfilter import HsvFilter
from threading import Thread
from agent import IronAgent
from oreDetection import OreDetection
from state import State

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Will show a window with the bots vision
SHOW_VISION = True
# Will enable the functionality of taking pictures for training data
GET_DATA = False

# To draw rectangles on the screenshots
ironOreVision = Vision(None)

# initialize the WindowCapture class
wincap = WinCap('BlueStacks', HsvFilter(6,119,0,12,147,96,0,0,0,0), ironOreVision)

# initialize the detector
detector = OreDetection('cascadeModel.xml')

# initialize the agent
agent = IronAgent((wincap.offsetX, wincap.offsetY), (wincap.w, wincap.h))

# Start the worker threads
wincap.start()
detector.start()
agent.start()

# Start main loop
loop_time = time()
print("Starting session...")
while(True):

    if wincap.screenshot is None:
        continue

    # give the detector a processed screenshot
    detector.update(wincap.processedScreenshot)

    if agent.state == State.INITIALIZING:
        targets = ironOreVision.returnCenterPos(detector.rectList)
        agent.updateTargets(targets)

    elif agent.state == State.SEARCHING:
        targets = ironOreVision.returnCenterPos(detector.rectList)
        agent.updateTargets(targets)

    elif agent.state == State.MINING:
        targets = ironOreVision.returnCenterPos(detector.rectList)
        agent.updateTargets(targets)

    if SHOW_VISION:
        screen = ironOreVision.displayRectangles(wincap.screenshot, detector.rectList)
        cv.imshow('Vision', screen)

    # Press q to exit loop and stop all threads
    key = cv.waitKey(1)
    if key == ord('q'):
        wincap.stop()
        detector.stop()
        agent.stop()
        cv.destroyAllWindows()
        break
    elif key == ord('p') and GET_DATA:
        cv.imwrite(f'positives/{loop_time}.jpg', wincap.screenshot)
    elif key == ord('f') and GET_DATA:
        cv.imwrite(f'negatives/{loop_time}.jpg', wincap.screenshot)

print("Stopped the session...")
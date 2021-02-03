import cv2 as cv
import pyautogui
from time import sleep, time
from threading import Thread, Lock
from math import sqrt
from random import uniform
from state import State

class IronAgent:
    # variables...
    START_UP_SECONDS = 5
    # Apparently it is not saved as a function, it will keep the random value.
    # Every time mining seconds are used, it will manually be changed.
    MINING_SECONDS = round(uniform(5,8), 2)

    running = False
    lock = None

    state = None
    targets = []
    timestamp = None
    windOffset = (0,0)
    windowW = 0
    windowH = 0


    def __init__(self, offSet, winSize):
        self.lock = Lock()

        self.windOffset = offSet
        self.windowW = winSize[0]
        self.windowH = winSize[1]

        self.state = State.INITIALIZING
        self.timestamp = time()

    def targetsOrderedByDist(self, targets):
        # https://stackoverflow.com/a/30636138/4655368
        def pythagorean_distance(pos):
            myPos = (self.windowW / 2, self.windowH / 2)
            return sqrt((pos[0] - myPos[0])**2 + (pos[1] - myPos[1])**2)
        targets.sort(key=pythagorean_distance)

        return targets

    def clickClosestTarget(self):
        # Create list of all targets, ordered by distance according to pythagorean dist
        targetList = self.targetsOrderedByDist(self.targets)

        # TODO
        # For every target in target list, we couuld check for tooltip in top left corner
        # to be absolutely sure it is a match. But this will not be prioritized.
        # Could be done by hovering position, using cv.matchTemplate with tooltip and screenshot
        # and use cv.minMaxLoc to see if the max_val is in some given threshold range.

        if (len(targetList) > 0):
            objPos = targetList[0]

            screenX, screenY = self.getScreenPositions(objPos)

            # We could start by moving the mouse to the location and then click, 
            # but this is not necessary as bluestack does not (should not) track mousemovement
            pyautogui.click(x=screenX, y=screenY)
        return True


    def getScreenPositions(self, pos):
        return (pos[0] + self.windOffset[0], pos[1] + self.windOffset[1])

    def updateTargets(self, targets):
        self.lock.acquire()
        self.targets = targets
        self.lock.release()

    def start(self):
        self.running = True
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            if self.state == State.INITIALIZING:
                if time() > self.timestamp + self.START_UP_SECONDS:
                    self.lock.acquire()
                    self.state = State.SEARCHING
                    self.lock.release()
            
            elif self.state == State.SEARCHING:
                searchSuccess = self.clickClosestTarget()

                if searchSuccess:
                    self.lock.acquire()
                    self.timestamp = time()
                    self.state = State.MINING
                    self.lock.release()
                else:
                    pass
            
            elif self.state == State.MINING:
                if time() > self.timestamp + self.MINING_SECONDS:
                    self.lock.acquire()
                    self.state = State.SEARCHING
                    self.lock.release()

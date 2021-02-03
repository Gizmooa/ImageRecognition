import cv2 as cv
from threading import Thread, Lock

class OreDetection:
    running = False
    lock = None
    rectList = []

    cascade = None
    screenshot = None

    def __init__(self, modelPath):
        self.lock = Lock()
        self.cascade = cv.CascadeClassifier(modelPath)

    def update(self, shot):
        self.lock.acquire()
        self.screenshot = shot
        self.lock.release()

    def start(self):
        self.running = True
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            if not self.screenshot is None:
                rectList = self.cascade.detectMultiScale(self.screenshot)

                self.lock.acquire()
                self.rectList = rectList
                self.lock.release()
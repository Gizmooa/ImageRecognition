import numpy as np
import win32gui, win32ui, win32con
from threading import Thread, Lock

class WinCap:
    running = True
    lock = None
    screenshot = None
    processedScreenshot = None
    vision = None
    hsvFilter = None

    hwnd = None
    offsetX = 0
    offsetY = 0
    croppedX = 0
    croppedY = 0
    w = 0
    h = 0

    def __init__(self, windowName, hsvFilter, vision):
        self.lock = Lock()

        self.hwnd = win32gui.FindWindow(None, windowName)
        if not self.hwnd:
            raise Exception('[-] No window given, or found.')
        
        self.vision = vision
        self.hsvFilter = hsvFilter

        # get the window size
        windowRectangle = win32gui.GetWindowRect(self.hwnd)
        self.w = windowRectangle[2] - windowRectangle[0]
        self.h = windowRectangle[3] - windowRectangle[1]

        # account for the window border and titlebar and cut them off
        boarderPixels = 0
        titlePixels = 40
        # -60 to remove the bluestacks menu bar.
        self.w = self.w - (boarderPixels * 2) - 60
        self.h = self.h - titlePixels - boarderPixels
        self.croppedX = boarderPixels
        self.croppedY = titlePixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offsetX = windowRectangle[0] + self.croppedX
        self.offsetY = windowRectangle[1] + self.croppedY

    # Simply takes a screenshot, and returns the image
    def getScreenshot(self):
        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.croppedX, self.croppedY), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        # dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() 
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[...,:3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img

    # Lists the name of all windows open, used to find the correct name
    # to pass to this class.
    def listWindowNames(self):
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # Given a position, return position on the reffered window
    def getScreenPosition(self, pos):
        return (pos[0] + self.offsetX, pos[1] + self.offsetY)

    def start(self):
        self.running = True
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        self.running = False

    # Will keep updating a normal screenshot, and a processed screenshot
    def run(self):
        while self.running:
            screen = self.getScreenshot()
            procScreen = self.vision.applyHSVFilter(screen, self.hsvFilter)
            self.lock.acquire()
            self.screenshot = screen
            self.processedScreenshot = procScreen
            self.lock.release()

import cv2 as cv
import numpy as np
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter

class Vision:
    TRACKBAR_WINDOW = "Trackbar"


    needle = None
    needleW = 0
    needleH = 0
    method = None

    def __init__(self, needlePath, method=cv.TM_CCOEFF_NORMED):
        self.method = method
        self.needle = cv.imread(needlePath, self.method)

        self.needleW = self.needle.shape[1]
        self.needleH = self.needle.shape[0]


    '''
    Finds needle images in haystack, which in this case will be
    screenshots primarily taken in real time. 
    '''
    def findObject(self, haystack, threshold = 0.5):
        matches = cv.matchTemplate(haystack, self.needle, self.method)

        locs = np.where(matches >= threshold)
        prettyLocs = list(zip(*locs[::-1]))

        # if we found no results, return now. this reshape of the empty array allows us to 
        # concatenate together results without causing an error
        if not prettyLocs:
            return np.array([], dtype=np.int32).reshape(0, 4)

        rectList = []
        for loc in prettyLocs:
            rect = [int(loc[0]), int(loc[1]), self.needleW, self.needleH]
            # If there is only one rectangle drawn on object, groupRectangle will remove that entry.
            # Simply add two entries for each rectangle to fix this.
            rectList.append(rect)
            rectList.append(rect)
        rectList, weights = cv.groupRectangles(rectList, groupThreshold=1, eps=0.5)

        return rectList


    def return_cen_pos(self, rectList):
        points = []
        # Loop over all the rectangles
        for (x, y, w, h) in rectList:
            # Determine the center position
            centerX = x + int(w/2)
            centerY = y + int(h/2)

            points.append((centerX, centerY))

        return points

    def displayRectangles(self, haystack, rectList):
        line_color = (0, 255, 0)
        line_type = cv.LINE_8

        for (x,y,w,h) in rectList:
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # Draw the box
            cv.rectangle(haystack, top_left, bottom_right, color=line_color, 
                        lineType=line_type, thickness=1)
        return haystack


    def displayCrossHair(self, haystack, points):
        marker_color = (255, 0, 255)
        marker_type = cv.MARKER_CROSS
        for (centerX, centerY) in points:
            cv.drawMarker(haystack, (centerX, centerY), 
                        color=marker_color, markerType=marker_type, 
                        markerSize=20, thickness=1)
        return haystack


    def init_control_gui(self):
        cv.namedWindow(self.TRACKBAR_WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.TRACKBAR_WINDOW, 350, 700)

        # required callback. we'll be using getTrackbarPos() to do lookups
        # instead of using the callback.
        def nothing(position):
            pass

        # create trackbars for bracketing.
        # OpenCV scale for HSV is H: 0-179, S: 0-255, V: 0-255
        cv.createTrackbar('HMin', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('HMax', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        # Set default value for Max HSV trackbars
        cv.setTrackbarPos('HMax', self.TRACKBAR_WINDOW, 179)
        cv.setTrackbarPos('SMax', self.TRACKBAR_WINDOW, 255)
        cv.setTrackbarPos('VMax', self.TRACKBAR_WINDOW, 255)

        # trackbars for increasing/decreasing saturation and value
        cv.createTrackbar('SAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('SSub', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VSub', self.TRACKBAR_WINDOW, 0, 255, nothing)

        # trackbars for edge creation
        cv.createTrackbar('KernelSize', self.TRACKBAR_WINDOW, 1, 30, nothing)
        cv.createTrackbar('ErodeIter', self.TRACKBAR_WINDOW, 1, 5, nothing)
        cv.createTrackbar('DilateIter', self.TRACKBAR_WINDOW, 1, 5, nothing)
        cv.createTrackbar('Canny1', self.TRACKBAR_WINDOW, 0, 200, nothing)
        cv.createTrackbar('Canny2', self.TRACKBAR_WINDOW, 0, 500, nothing)
        # Set default value for Canny trackbars
        cv.setTrackbarPos('KernelSize', self.TRACKBAR_WINDOW, 5)
        cv.setTrackbarPos('Canny1', self.TRACKBAR_WINDOW, 100)
        cv.setTrackbarPos('Canny2', self.TRACKBAR_WINDOW, 200)

    def get_hsv_filter_from_controls(self):
        # Get current positions of all trackbars
        hsv_filter = HsvFilter()
        hsv_filter.hMin = cv.getTrackbarPos('HMin', self.TRACKBAR_WINDOW)
        hsv_filter.sMin = cv.getTrackbarPos('SMin', self.TRACKBAR_WINDOW)
        hsv_filter.vMin = cv.getTrackbarPos('VMin', self.TRACKBAR_WINDOW)
        hsv_filter.hMax = cv.getTrackbarPos('HMax', self.TRACKBAR_WINDOW)
        hsv_filter.sMax = cv.getTrackbarPos('SMax', self.TRACKBAR_WINDOW)
        hsv_filter.vMax = cv.getTrackbarPos('VMax', self.TRACKBAR_WINDOW)
        hsv_filter.sAdd = cv.getTrackbarPos('SAdd', self.TRACKBAR_WINDOW)
        hsv_filter.sSub = cv.getTrackbarPos('SSub', self.TRACKBAR_WINDOW)
        hsv_filter.vAdd = cv.getTrackbarPos('VAdd', self.TRACKBAR_WINDOW)
        hsv_filter.vSub = cv.getTrackbarPos('VSub', self.TRACKBAR_WINDOW)
        return hsv_filter

    
    def apply_hsv_filter(self, haystack, hsv_filter=None):
        # convert image to HSV from BGR
        hsv = cv.cvtColor(haystack, cv.COLOR_BGR2HSV)

        # if we haven't been given a defined filter, use the filter values from the GUI
        if not hsv_filter:
            hsv_filter = self.get_hsv_filter_from_controls()

        # add/subtract saturation and value
        h, s, v = cv.split(hsv)
        s = self.shift_channel(s, hsv_filter.sAdd)
        s = self.shift_channel(s, -hsv_filter.sSub)
        v = self.shift_channel(v, hsv_filter.vAdd)
        v = self.shift_channel(v, -hsv_filter.vSub)
        # Merge to a single image
        hsv = cv.merge([h, s, v])

        # Set minimum and maximum HSV values to display
        lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])
        # Apply the thresholds
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(hsv, hsv, mask=mask)

        # convert back to BGR for imshow() to display it properly
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        return img

    def get_edge_filter_from_controls(self):
        # Get current positions of all trackbars
        edge_filter = EdgeFilter()
        edge_filter.kernelSize = cv.getTrackbarPos('KernelSize', self.TRACKBAR_WINDOW)
        edge_filter.erodeIter = cv.getTrackbarPos('ErodeIter', self.TRACKBAR_WINDOW)
        edge_filter.dilateIter = cv.getTrackbarPos('DilateIter', self.TRACKBAR_WINDOW)
        edge_filter.canny1 = cv.getTrackbarPos('Canny1', self.TRACKBAR_WINDOW)
        edge_filter.canny2 = cv.getTrackbarPos('Canny2', self.TRACKBAR_WINDOW)
        return edge_filter


    def apply_edge_filter(self, original_image, edge_filter=None):
        # if we haven't been given a defined filter, use the filter values from the GUI
        if not edge_filter:
            edge_filter = self.get_edge_filter_from_controls()

        kernel = np.ones((edge_filter.kernelSize, edge_filter.kernelSize), np.uint8)
        eroded_image = cv.erode(original_image, kernel, iterations=edge_filter.erodeIter)
        dilated_image = cv.dilate(eroded_image, kernel, iterations=edge_filter.dilateIter)

        # canny edge detection
        result = cv.Canny(dilated_image, edge_filter.canny1, edge_filter.canny2)

        # convert single channel image back to BGR
        img = cv.cvtColor(result, cv.COLOR_GRAY2BGR)

        return img


    # A try of doing ORB Feature matching
    def match_keypoints(self, original_image, patch_size=16):
        # How many matches to determine if it is a good match
        min_match_count = 5

        orb = cv.ORB_create(edgeThreshold=0, patchSize=patch_size)
        keypoints_needle, descriptors_needle = orb.detectAndCompute(self.needle, None)
        orb2 = cv.ORB_create(edgeThreshold=0, patchSize=patch_size, nfeatures=1000)
        keypoints_haystack, descriptors_haystack = orb2.detectAndCompute(original_image, None)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, 
                table_number=6,
                key_size=12,    
                multi_probe_level=1)

        search_params = dict(checks=50)

        try:
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors_needle, descriptors_haystack, k=2)
        except cv.error:
            return None, None, [], [], None

        # store all the good matches as per Lowe's ratio test.
        good = []
        points = []

        for pair in matches:
            if len(pair) == 2:
                if pair[0].distance < 0.7*pair[1].distance:
                    good.append(pair[0])

        if len(good) > min_match_count:
            print('match %03d, kp %03d' % (len(good), len(keypoints_needle)))
            for match in good:
                points.append(keypoints_haystack[match.trainIdx].pt)
            #print(points)
        
        return keypoints_needle, keypoints_haystack, good, points

    def centeroid(self, point_list):
        point_list = np.asarray(point_list, dtype=np.int32)
        length = point_list.shape[0]
        sum_x = np.sum(point_list[:, 0])
        sum_y = np.sum(point_list[:, 1])
        return [np.floor_divide(sum_x, length), np.floor_divide(sum_y, length)]

    # apply adjustments to an HSV channel
    # Makes sure when adding 10 to one with 255, it will not overflow
    # https://stackoverflow.com/questions/49697363/shifting-hsv-pixel-values-in-python-using-numpy
    def shift_channel(self, c, amount):
        if amount > 0:
            lim = 255 - amount
            c[c >= lim] = 255
            c[c < lim] += amount
        elif amount < 0:
            amount = -amount
            lim = amount
            c[c <= lim] = 0
            c[c > lim] -= amount
        return c
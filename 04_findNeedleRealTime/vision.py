import cv2 as cv
import numpy as np

class Vision:
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
    def findObject(self, haystack, threshold = 0.5, draw='rectangle'):
        matches = cv.matchTemplate(haystack, self.needle, self.method)

        locs = np.where(matches >= threshold)
        prettyLocs = list(zip(*locs[::-1]))

        rectList = []
        for loc in prettyLocs:
            rect = [int(loc[0]), int(loc[1]), self.needleW, self.needleH]
            # If there is only one rectangle drawn on object, groupRectangle will remove that entry.
            # Simply add two entries for each rectangle to fix this.
            rectList.append(rect)
            rectList.append(rect)
        rectList, weights = cv.groupRectangles(rectList, groupThreshold=1, eps=0.5)

        points = []
        if (len(rectList)):
            line_color = (0, 255, 0)
            line_type = cv.LINE_8
            marker_color = (255, 0, 255)
            marker_type = cv.MARKER_CROSS

            # Loop over all the rectangles
            for (x, y, w, h) in rectList:

                # Determine the center position
                center_x = x + int(w/2)
                center_y = y + int(h/2)

                points.append((center_x, center_y))

                if draw == 'rectangles':
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    # Draw the box
                    cv.rectangle(haystack, top_left, bottom_right, color=line_color, 
                                lineType=line_type, thickness=2)
                elif draw == 'points':
                    # Draw the center point
                    cv.drawMarker(haystack, (center_x, center_y), 
                                color=marker_color, markerType=marker_type, 
                                markerSize=20, thickness=2)

            if draw:
                cv.imshow('Vision', haystack)
                #cv.waitKey()
                #cv.imwrite('result_click_point.jpg', haystack_img)

        return points



        
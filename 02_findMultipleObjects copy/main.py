import cv2 as cv 
import numpy as np
import math, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(round(stepper * number, digits * 3)) / stepper


def findObjects(needle, haystack, threshold=0.5, method=cv.TM_CCOEFF_NORMED, draw='rectangles'):
    haystack = cv.imread(haystack, method)
    needle = cv.imread(needle, method)

    needleW = needle.shape[1]
    needleH = needle.shape[0]

    # returns a multidimension array, consisting of how bright the pixels are / confidence
    # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
    matches = cv.matchTemplate(haystack, needle, method)

    locs = np.where(matches >= threshold)
    prettyLocs = list(zip(*locs[::-1]))

    rectList = []
    for loc in prettyLocs:
        rect = [int(loc[0]), int(loc[1]), needleW, needleH]
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
            cv.imshow('Matches', haystack)
            cv.waitKey()
            #cv.imwrite('result_click_point.jpg', haystack_img)

    return points

needle = 'needle.jpg'
haystack = 'haystack.jpg'
locs = findObjects(needle, haystack, 0.55, cv.TM_CCOEFF_NORMED, draw='rectangles')
locs = findObjects(needle, haystack, 0.55, cv.TM_CCOEFF_NORMED, draw='points')
print(locs)

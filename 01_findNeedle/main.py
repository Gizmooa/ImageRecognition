import cv2 as cv 
import numpy as np
import math


def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(round(stepper * number, digits * 3)) / stepper

# cv.IMREAD_REDUCED_COLOR_2 to downscale
haystack = cv.imread('01_findNeedle/haystack.jpg', cv.IMREAD_UNCHANGED)
needle = cv.imread('01_findNeedle/needle.jpg', cv.IMREAD_UNCHANGED)

# returns a multidimension array, consisting of how bright the pixels are / confidence
matches = cv.matchTemplate(haystack, needle, cv.TM_CCOEFF_NORMED)

# returns the best match location
minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(matches)

print('Best match top left position: %s' % str(maxLoc))
print('Best match confidence: %s' % maxVal)

threshold = 0.8
if maxVal >= threshold:
    print(f"[+] We've found the needle with confidence {truncate(maxVal*100, 3)} percent")
    
    # dimensions of the needle img
    needleW = needle.shape[1]
    needleH = needle.shape[0]
    line_color = (0,255,0)
    line_type = cv.LINE_4
    
    topLeft = maxLoc
    bottomRight = (topLeft[0] + needleW, topLeft[1] + needleH)
    cv.rectangle(haystack, topLeft, bottomRight, color=line_color, thickness=1, lineType=line_type)
    
    # imwrite to save to a file, imshow to simply show
    #cv.imwrite('matching.jpg', haystack)
    cv.imshow('Matching', haystack)
    cv.waitKey()

else:
    print(f"[-] We've not found the needle with threshold {threshold}")




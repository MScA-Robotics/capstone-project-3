import cv2
import numpy as np
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import pdb


def take_picture(filename):
    camera = PiCamera()
    try:
       camera.capture(filename)
    finally:
       camera.close()


def convexHullPointingUp(ch):
    pointsAboveCenter, poinstBelowCenter = [], []

    x, y, w, h = cv2.boundingRect(ch)
    aspectRatio = w / h

    if aspectRatio < 0.8:
        verticalCenter = y + h / 2

        for point in ch:
            if point[0][1] < verticalCenter:
                pointsAboveCenter.append(point)
            elif point[0][1] >= verticalCenter:
                poinstBelowCenter.append(point)

        leftX = poinstBelowCenter[0][0][0]
        rightX = poinstBelowCenter[0][0][0]
        for point in poinstBelowCenter:
            if point[0][0] < leftX:
                leftX = point[0][0]
            if point[0][0] > rightX:
                rightX = point[0][0]

        for point in pointsAboveCenter:
            if (point[0][0] < leftX) or (point[0][0] > rightX):
                return False

    else:
        return False

    return True



def findCone(colorBounds = [([119, 82, 29],[135, 225, 96])], imageSource = "camera"):
    """"
    Input: 
        - colorBounds: as a list/touple of lists OR a nested list of touple of lists (from Raghav's get_hsv_boundaries() )
        - imageSource: "camera" for using picamera OR path to saved image.
    Output: Horizontal position of the center of the cone in the frame (as a decimal between (0,1)) OR False if no cone found
    Dependencies: Numpy, cv2, picamera, time
    Limitations: picamera can only run on the raspberry pi itself.  Comment out the picamera imports and use saved images if testing on a different platform
    """

    #_____imgOriginal
    if imageSource == "camera":
        with PiCamera() as camera:
            res = (640, 480)
            camera.resolution = res
            camera.framerate = 24
            time.sleep(0.1)
            rawCapture = PiRGBArray(camera, size = res)
            camera.capture(rawCapture, 'bgr')
            imgOriginal = rawCapture.array
    else: 
        imgOriginal = cv2.imread(imageSource)
        
    #_____imgHSV
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    width = imgHSV.shape[1]

    #_____colorThrshold
    if any((isinstance(j, list) | isinstance(j, tuple)) for i in colorBounds for j in i):
        colorBounds = colorBounds[0]
    lowerColor = np.array(colorBounds[0])
    upperColor = np.array(colorBounds[1])
    imgThresh = cv2.inRange(imgHSV, lowerColor, upperColor)

    #_____imgThreshSmoothed
    kernel = np.ones((3, 3), np.uint8)
    imgEroded = cv2.erode(imgThresh, kernel, iterations=1)
    imgDilated = cv2.dilate(imgEroded, kernel, iterations=1)
    imgThreshSmoothed = cv2.GaussianBlur(imgDilated, (3, 3), 0)

    #_____imgCanny
    imgCanny = cv2.Canny(imgThreshSmoothed, 80, 160)

    #_____imgContours
    contours, _ = cv2.findContours(np.array(imgCanny), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    approxContours = []

    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed=True)
        approxContours.append(approx)


    #_____imgAllConvexHulls
    allConvexHulls = []

    for ac in approxContours:
        allConvexHulls.append(cv2.convexHull(ac))

    #_____imgConvexHulls3To10
    convexHull3To10 = []

    for ch in allConvexHulls:
        if 3 <= len(ch) <= 10:
            convexHull3To10.append(cv2.convexHull(ch))


    #imgTrafficCones

    cones = []
    bounding_Rects = []

    for ch in convexHull3To10:
        if convexHullPointingUp(ch):
            cones.append(ch)
            #boundingRect returns (topleft_x,topleft_y,width,height)
            rect = cv2.boundingRect(ch)
            bounding_Rects.append(rect)

    #horizontal centers
    centers = []
    for rect in bounding_Rects:
        center = rect[0]+rect[2]-rect[2]/2
        centers.append(center)
    #return horizontal position of (first) found cone
    try:
        return round(centers[0]/width, 5)
    except IndexError:
        return False

#The chose_cone definition is not yet  completed.
def findcone_mod(color, cones):
    bounding_Rects = []
    centers = []
    #print(cones)
    #print(color)
                
    for cone in [cones]:
        print(cone)
        if cone[1][0] == color:
            rect= cone[0][0][0]
            bounding_Rects.append(rect)

    #horizontal centers
    centers = []
    for rect in bounding_Rects:
        center = rect[0]+rect[2]-rect[2]/2
        centers.append(center)
    #return horizontal position of (first) found cone
    try:
        #return round(centers[0]/width,5)
        return round(centers[0]/3280, 5)
    except IndexError:
        return False


#!/usr/bin/env python
# coding: utf-8


# import the necessary packages
import numpy as np
import cv2



def get_hsv_boundaries(imgPath):
    """"
    Input: path to the image
    Output: Boundary of HSV values in the format [([lower_H, lower_S, lower_V], [upper_H, upper_S, upper_V])]
    This function will present the user with a window and sliders to adjust the values of HSV boundaries
    Once the user is satisfied with the slider settings they can press escape key to get a return value with boundaries

    Dependencies: Numpy, cv2
    """

    if not imgPath:
        return None

    #Read the image into an array
    image = cv2.imread(imgPath)

    if image is None:
        return None

    resized_image = cv2.resize(image, (200, 200)) #resize image
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV) #convert to hsv format

    def nothing(x):
        pass

    #Create cv2 window with trackbars
    cv2.namedWindow('image')
    cv2.createTrackbar('lower_H', 'image', 128, 255, nothing)
    cv2.createTrackbar('lower_S', 'image', 41, 255, nothing)
    cv2.createTrackbar('lower_V', 'image', 41, 255, nothing)

    cv2.createTrackbar('upper_H', 'image', 255, 255, nothing)
    cv2.createTrackbar('upper_S', 'image', 255, 255, nothing)
    cv2.createTrackbar('upper_V', 'image', 255, 255, nothing)
    print("HSV slider window is presented")
    print("Press escape to return the HSV boundary settings")
    while (1):
        lw_h = cv2.getTrackbarPos('lower_H', 'image')
        lw_s = cv2.getTrackbarPos('lower_S', 'image')
        lw_v = cv2.getTrackbarPos('lower_V', 'image')

        up_h = cv2.getTrackbarPos('upper_H', 'image')
        up_s = cv2.getTrackbarPos('upper_S', 'image')
        up_v = cv2.getTrackbarPos('upper_V', 'image')

        boundaries = [
            ([lw_h, lw_s, lw_v], [up_h, up_s, up_v])
        ]
        #print(boundaries)

        lower = boundaries[0][0]
        upper = boundaries[0][1]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # print(lower, upper)
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(resized_image, lower, upper)
        output = cv2.bitwise_and(resized_image, resized_image, mask=mask)

        # show the images
        cv2.imshow("image", np.hstack([resized_image, output]))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            #print('Boundaries for HSV are:')
            #print(boundaries)
            print("Exiting HSV slider window")
            break

    return boundaries

#image = cv2.imread("data/cone1.jpeg")
#image = cv2.imread("data/cone2.jpeg")
#image = cv2.imread("data/coneDark.jpeg")
image = "data/vertical_image.jpeg"
print('Boundaries for HSV are:' , get_hsv_boundaries(image))
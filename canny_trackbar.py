# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 00:08:01 2017

@author: SROY
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:31:17 2015
Canny Edge Detection tool with trackbars for varying thresholds.
@author: Johnny
"""

import cv2

path = 'C:/Users/SROY/Desktop/Courses/CS513/TestImages/393408638.jpg'
# this function is needed for the createTrackbar step downstream
def nothing(x):
    pass

# read the experimental image
img = cv2.imread(path, 0)
#Gausblur = cv2.GaussianBlur(imgTemp,(3,3),0)
# create trackbar for canny edge detection threshold changes
cv2.namedWindow('canny')
cv2.resizeWindow('canny', 300, 300)

# add ON/OFF switch to "canny"
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'canny', 0, 1, nothing)

# add lower and upper threshold slidebars to "canny"
cv2.createTrackbar('lower', 'canny', 0, 255, nothing)
cv2.createTrackbar('upper', 'canny', 0, 255, nothing)

# Infinite loop until we hit the escape key on keyboard
while(1):

    # get current positions of four trackbars
    lower = cv2.getTrackbarPos('lower', 'canny')
    upper = cv2.getTrackbarPos('upper', 'canny')
    s = cv2.getTrackbarPos(switch, 'canny')

    if s == 0:
        edges = img
    else:
        edges = cv2.Canny(img, lower, upper)

    # display images
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original', 300, 300)
    cv2.imshow('original', img)
    
    cv2.namedWindow('canny', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('canny', 300, 300)
    cv2.imshow('canny', edges)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:   # hit escape to quit
        break

#cv2.waitKey(0)
cv2.destroyAllWindows()

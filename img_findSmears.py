# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:24:03 2017

@author: SROY
"""

#path = 'C:\\Users\\SROY\\Desktop\\Courses\\CS513\\TestImages_Variance.jpg'
path = 'C:\\Users\\SROY\\Desktop\\Courses\\CS513\\TestImages\\393408638.jpg'
import cv2
import numpy as np

#from IPython.display import display, Image
#from scipy import ndimage
#image_data = ndimage.imread(path).astype(float)
#type(image_data)
#display(Image(image_data))

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

#Kernel function
kernel = np.ones((5,5),np.uint8)

#Gaussian Blur
Gausblur = cv2.GaussianBlur(img,(3,3),0)

#Edges using canny
edges = cv2.Canny(Gausblur, 10, 5)  #10,5

#Binary image
ret, threshBin = cv2.threshold(edges, 127, 255,cv2.THRESH_BINARY_INV)

#Erode Image
erosionImg = cv2.erode(threshBin,kernel,iterations = 2)  

#Dilate Image
dilateImg = cv2.dilate(erosionImg,kernel,iterations = 3)  #Image looks fine till here

# Find Contours 
image, contours, hierarchy=cv2.findContours(dilateImg, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# Draw Contour
cv2.drawContours(dilateImg, contours, -1, (0, 255, 0), 25)

#Remove false pos and false neg
opening = cv2.morphologyEx(dilateImg, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Invert Image
mask = 255 - closing
res = cv2.bitwise_and(img, img, mask = mask)

#
## Setup SimpleBlobDetector parameters.
#params = cv2.SimpleBlobDetector_Params()
#params.blobColor= 255
#params.filterByColor = True
#
## Change thresholds
#params.minThreshold = 200
#params.maxThreshold = 255
#
#detector = cv2.SimpleBlobDetector_create(params)
## Detect blobs.
#keypoints = detector.detect(dilateImg)
#im_with_keypoints = cv2.drawKeypoints(dilateImg, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


#Resize output and generate
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 300, 300)
cv2.imshow('img', img)


cv2.namedWindow('New Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('New Image', 300, 300)
cv2.imshow('New Image', res)

#Kill
cv2.waitKey(0)
cv2.destroyAllWindows()
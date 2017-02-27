# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:51:59 2017

@author: SROY
"""
path = 'C:/Users/SROY/Desktop/Courses/CS513/TestImages/393408638.jpg'
import cv2
import numpy as np
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
imgTemp = img

#Kernel function
kernel = np.ones((5,5),np.uint8)

#Erode Image
erosionImg = cv2.erode(img,kernel,iterations = 1)  

#Dilate Image
dilateImg = cv2.dilate(img,kernel,iterations = 1)

#Increase contrast
contrastImg = img * 3

#Laplace
laplacianImg = cv2.Laplacian(img,cv2.CV_64F)

#Sobel
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

#Edges
edges = cv2.Canny(img,100,200)

#Opening/Closing
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

#Median Blur
medianBlur = cv2.medianBlur(img,15)

#Gaussian Blur
Gausblur = cv2.GaussianBlur(img,(15,15),0)

#Averaging
kernelAv = np.ones((15,15),np.float32)/225
smoothAv = cv2.filter2D(img, -1, kernelAv)

#Binary image
ret, threshBin = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#Weighted average of multiple images
#weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

#Blob detector
#detector = cv2.SimpleBlobDetector_create()
#keypoints = detector.detect(cont)
#im_with_keypoints = cv2.drawKeypoints(cont, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Resize output and generate
#cv2.namedWindow('img', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('img', 300, 300)
#cv2.imshow('img', img)
#
#cv2.namedWindow('erosionImg', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('erosionImg', 300, 300)
#cv2.imshow('erosionImg', erosionImg)
#
#cv2.namedWindow('dilateImg', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('dilateImg', 300, 300)
#cv2.imshow('dilateImg', dilateImg)
#
#cv2.namedWindow('contrastImg', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('contrastImg', 300, 300)
#cv2.imshow('contrastImg', contrastImg)
#
#cv2.namedWindow('laplacianImg', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('laplacianImg', 300, 300)
#cv2.imshow('laplacianImg', laplacianImg)
#
#cv2.namedWindow('sobelx', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('sobelx', 300, 300)
#cv2.imshow('sobelx', sobelx)
#
#cv2.namedWindow('sobely', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('sobely', 300, 300)
#cv2.imshow('sobely', sobely)
#
#cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('edges', 300, 300)
#cv2.imshow('edges', edges)
#
#cv2.namedWindow('opening', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('opening', 300, 300)
#cv2.imshow('opening', opening)
#
#cv2.namedWindow('closing', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('closing', 300, 300)
#cv2.imshow('closing', closing)
#
#cv2.namedWindow('medianBlur', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('medianBlur', 300, 300)
#cv2.imshow('medianBlur', medianBlur)
#
#cv2.namedWindow('smoothAv', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('smoothAv', 300, 300)
#cv2.imshow('smoothAv', smoothAv)
#
#cv2.namedWindow('Gausblur', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Gausblur', 300, 300)
#cv2.imshow('Gausblur', Gausblur)

cv2.namedWindow('threshBin', cv2.WINDOW_NORMAL)
cv2.resizeWindow('threshBin', 300, 300)
cv2.imshow('threshBin', threshBin)

#Kill
cv2.waitKey(0)
cv2.destroyAllWindows()
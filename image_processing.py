# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
img = 'C:/Users/SROY/Desktop/Courses/CS513/TestImages/393408638.jpg'

# Display as ndimage using scipy
from IPython.display import display, Image
from scipy import ndimage
image_data = ndimage.imread(img).astype(float)
type(image_data)
display(Image(img))


# Display using skimage
from skimage import io, filters
img_array = io.imread(img)
type(img_array)
fil_arr = filters.gaussian_filter(img_array, sigma = 5)
plt.imshow(img_array, cmap='gray')

# Chelsea image
from skimage import data
color_image = data.chelsea()
print(color_image.shape)
plt.imshow(color_image);
 
# Test 513 code
path = 'C:/Users/SROY/Desktop/Courses/CS513/TestImages/393408638.jpg'
import cv2
import numpy as np
#import matplotlib.pyplot as plt #dont use matplotlib with opencv due to rgb and brg
#from IPython.display import display, Image
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

#Blob detector
#detector = cv2.SimpleBlobDetector_create()
#keypoints = detector.detect(cont)
#im_with_keypoints = cv2.drawKeypoints(cont, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Resize output and generate
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 300, 300)
cv2.imshow('img', img)

cv2.namedWindow('erosionImg', cv2.WINDOW_NORMAL)
cv2.resizeWindow('erosionImg', 300, 300)
cv2.imshow('erosionImg', erosionImg)

cv2.namedWindow('dilateImg', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dilateImg', 300, 300)
cv2.imshow('dilateImg', dilateImg)

#cv2.namedWindow('contrastImg', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('contrastImg', 300, 300)
cv2.imshow('contrastImg', contrastImg)

cv2.namedWindow('laplacianImg', cv2.WINDOW_NORMAL)
cv2.resizeWindow('laplacianImg', 300, 300)
cv2.imshow('laplacianImg', laplacianImg)

#Kill
cv2.waitKey(0)
cv2.destroyAllWindows()
#plt.imshow(keypoints)#, cmap = 'gray', interpolation = 'bicubic')


    
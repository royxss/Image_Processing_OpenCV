import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

###
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
plt.show()

###
cv2.imwrite('watchgray.png',img)

###
import numpy as np
import cv2


img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)

###############################################################

#You are encouraged to use your own image. As usual, our starting code can be something like:

import numpy as np
import cv2

img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)
#Next, we can start drawing, like:

cv2.line(img,(0,0),(150,150),(255,255,255),15)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#The cv2.line() takes the following parameters: where, start coordinates, end coordinates, color (bgr), line thickness.

#Alright, cool, let's get absurd with some more shapes. Next up, a rectangle:

cv2.rectangle(img,(15,25),(200,150),(0,0,255),15)
#The parameters here are the image, the top left coordinate, bottom right coordinate, color, and line thickness.

#How about a circle?

cv2.circle(img,(100,63), 55, (0,255,0), -1)
#The parameters here are the image/frame, the center of the circle, the radius, color, and then thickness. Notice we have a -1 for thickness. This means the object will actually be filled in, so we will get a filled in circle.

#Lines, rectangles, and circles are cool and all, but what if we want a pentagon, or octagon, or octdecagon?! No problem!

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# OpenCV documentation had this code, which reshapes the array to a 1 x 2. I did not 
# find this necessary, but you may:
#pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,255,255), 3)
#First, we name pts, short for points, as a numpy array of coordinates. Then, we use cv2.polylines to draw the lines. The parameters are as follows: where is the object being drawn to, the coordinates, should we "connect" the final and starting dot, the color, and again the thickness.

#The final thing you may want to do is write on the image. This can be done like so:

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV Tuts!',(0,130), font, 1, (200,255,155), 2, cv2.LINE_AA)
#Full code up to this point would be something like:

import numpy as np
import cv2

img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)
cv2.line(img,(0,0),(200,300),(255,255,255),50)
cv2.rectangle(img,(500,250),(1000,500),(0,0,255),15)
cv2.circle(img,(447,63), 63, (0,255,0), -1)
pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,255,255), 3)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV Tuts!',(10,500), font, 6, (200,255,155), 13, cv2.LINE_AA)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


##################################################################
import cv2
import numpy as np

img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)
#Now, we can reference specific pixels, like so:

px = img[55,55]
#Next, we could actually change a pixel:

img[55,55] = [255,255,255]
#Then re-reference:

px = img[55,55]
print(px)
#It should be different now. Next, we can reference an ROI, or Region of Image, like so:

px = img[100:150,100:150]
print(px)
#We can also modify the ROI, like this:

img[100:150,100:150] = [255,255,255]
#We can reference certain characteristics of our image:

print(img.shape)
print(img.size)
print(img.dtype)
#And we can perform operations, like:

watch_face = img[37:111,107:194]
img[0:74,0:87] = watch_face

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

###############################################################################
# Canny edge detection and gradients

import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while(1):

    # Take each frame
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    laplacian = cv2.Laplacian(frame,cv2.CV_64F)
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)

    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely',sobely)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('Original',frame)
    edges = cv2.Canny(frame,100,200)
    cv2.imshow('Edges',edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

#########################################################
# Morphological transformation and filters

These tend to come in pairs. The first pair we're going to talk about is Erosion and Dilation. Erosion is where we will "erode" the edges. The way these work is we work with a slider (kernel). We give the slider a size, let's say 5 x 5 pixels. What happens is we slide this slider around, and if all of the pixels are white, then we get white, otherwise black. This may help eliminate some white noise. The other version of this is Dilation, which basically does the opposite: Slides around, if the entire area isn't black, then it is converted to white. Here's an example:

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(mask,kernel,iterations = 1)

    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('Erosion',erosion)
    cv2.imshow('Dilation',dilation)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()


The next pair is "opening" and "closing." The goal with opening is to remove "false positives" so to speak. Sometimes, in the background, you get some pixels here and there of "noise." The idea of "closing" is to remove false negatives. Basically this is where you have your detected shape, like our hat, and yet you still have some black pixels within the object. Closing will attempt to clear that up.

cap = cv2.VideoCapture(1)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    kernel = np.ones((5,5),np.uint8)
    
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('Opening',opening)
    cv2.imshow('Closing',closing)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

###########################################################
#Blurring and smoothing

As you can see, we have a lot of black dots where we'd prefer red, and a lot of other colored dots scattered about. We can use various blurring and smoothing techniques to attempt to remedy this a bit. We can start with some familiar code:

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
	
Now, let's apply a simple smoothing, where we do a sort of averaging per block of pixels. In our case, let's do a 15 x 15 square, which means we have 225 total pixels.

    kernel = np.ones((15,15),np.float32)/225
    smoothed = cv2.filter2D(res,-1,kernel)
    cv2.imshow('Original',frame)
    cv2.imshow('Averaging',smoothed)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

This one is simple enough, but the result sacrifices a lot of granularity. Next, let's try some Gaussian Blurring:

    blur = cv2.GaussianBlur(res,(15,15),0)
    cv2.imshow('Gaussian Blurring',blur)

Another option is what is called Median Blur:

    median = cv2.medianBlur(res,15)
    cv2.imshow('Median Blur',median)

Finally, another option is the bilateral blur:

    bilateral = cv2.bilateralFilter(res,15,75,75)
    cv2.imshow('bilateral Blur',bilateral)


########################################################
#Image operations
import cv2
import numpy as np

img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)

Now, we can reference specific pixels, like so:

px = img[55,55]
Next, we could actually change a pixel:

img[55,55] = [255,255,255]
Then re-reference:

px = img[55,55]
print(px)
It should be different now. Next, we can reference an ROI, or Region of Image, like so:

px = img[100:150,100:150]
print(px)
We can also modify the ROI, like this:

img[100:150,100:150] = [255,255,255]
We can reference certain characteristics of our image:

print(img.shape)
print(img.size)
print(img.dtype)

And we can perform operations, like:

watch_face = img[37:111,107:194]
img[0:74,0:87] = watch_face

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

######################################################

#Image arithmatic and logic

First, let's see what a simple addition will do:

import cv2
import numpy as np

# 500 x 250
img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainsvmimage.png')

add = img1+img2

cv2.imshow('add',add)
cv2.waitKey(0)
cv2.destroyAllWindows()

It is unlikely you will want this sort of messy addition. OpenCV has an "add" method, let's see what that does, replacing the previous "add" with:

add = cv2.add(img1,img2)

Probably not the ideal here either. We can see that much of the image is very "white." This is because colors are 0-255, where 255 is "full light." Thus, for example: (155,211,79) + (50, 170, 200) = 205, 381, 279...translated to (205, 255,255).

Next, we can add images, and have each carry a different "weight" so to speak. Here's how that might work:

import cv2
import numpy as np

img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainsvmimage.png')

weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
cv2.imshow('weighted',weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()

Now, we can take this logo, and place it on the original image. That would be pretty easy (basically using the same-ish code we used in the previous tutorial where we replaced the Region of Image (ROI) with a new one), but what if we just want the logo part, and not the white background? We can use the same principle as we had used before for the ROI replacement, but we need a way to "remove" the background of the logo, so that the white is not needlessly blocking more of the background image. First I will show the full code, and then explain:

import cv2
import numpy as np

# Load two images
img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainlogo.png')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# add a threshold
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
A decent amount happened here, and a few new things showed up. The first thing we see that is new, is the application of a threshold: ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV).

We will be covering thresholding more in the next tutorial, so stay tuned for the specifics, but basically the way it works is it will convert all pixels to either black or white, based on a threshold value. In our case, the threshold is 220, but we can use other values, or even dynamically choose one, which is what the ret variable can be used for. Next, we see: mask_inv = cv2.bitwise_not(mask). This is a bitwise operation. Basically, these operators are very similar to the typical ones from python, except for one, but we wont be touching it anyway here. In this case, the invisible part is where the black is. Then, we can say that we want to black out this area in the first image, and then take image 2 and replace it's contents in that empty spot.


	

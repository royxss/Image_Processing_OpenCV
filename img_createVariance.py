# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:52:14 2017

@author: SROY
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit

def findWeightAv(image_files):
    first = os.path.join(folder, image_files[0])
    #first = first.replace('\\','/')
    img = cv2.imread(first, cv2.IMREAD_GRAYSCALE)
    img_sum = np.zeros_like(img, dtype=np.float64)
    img_sqr_sum = np.zeros_like(img, dtype=np.float64)
    
    for name in image_files:
        image_file = os.path.join(folder, name)
        #image_file = image_file.replace('\\','/')
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        img_sum += img
        img_sqr_sum += img ** 2
    #print(img_sum)
    #print(img_sqr_sum)
    variance = img_sqr_sum / len(image_files) - ((img_sum / len(image_files)) ** 2)
    return variance

def main(folder):
    image_files = os.listdir(folder)
    min_var = 500
    white_fill = 255
    start = timeit.timeit()
    Var = findWeightAv(image_files)
    
    lens_smear = np.zeros_like(Var, np.uint)
    lens_smear[np.where(Var > min_var)] = white_fill
    
    end = timeit.timeit()
    print("Time taken : ", end - start)
    print(type(lens_smear))
    print(lens_smear)
    plt.imshow(lens_smear)
    plt.colorbar()
    plt.show()
    basename = os.path.basename(folder)
    cv2.imwrite(os.path.dirname(folder) + "\\%s_Variance.jpg" % basename, lens_smear)
    #np.save(os.path.dirname(folder) + "\\%s_Variance" % basename, Var)

folder = 'C:\\Users\\SROY\\Desktop\\Courses\\CS513\\TestImages'
#folder = 'C:\\Users\\SROY\\Desktop\\Courses\\CS513\\DataHw1\\sample_drive\\cam_0' 

if __name__ == '__main__':   
    main(folder)

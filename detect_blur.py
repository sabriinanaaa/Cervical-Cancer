# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:12:21 2022

@author: Sabrina
"""
import os
import cv2
from imutils import paths
from PIL import Image as im
from pyimagesearch.detect_blur_fft import detect_blur_fft

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

args = {"threshold": 100}

typeimgPath = 'train\\Type_1'
ouputPath = 'train_blur_image\\Type_1'
ouputPath_resized = 'train_nonblur\\Type_1'

print("[INFO] loading images")
typeimgPaths = list(paths.list_images(typeimgPath))

means=[]
data=[]
fms=[]

for imagePath in typeimgPaths:

    print("[progress]{}".format(imagePath))
    image = cv2.imread(imagePath)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (500, 500))
    fm = variance_of_laplacian(gray_resized)

  
    if fm < args["threshold"]:
        print('fm=',fm)
        fms.append(fm)
        (mean, blurry) = detect_blur_fft(gray_resized, size=60, thresh=0, vis=False)
        if mean <= 0:
            print('mean=',mean)
            means.append(mean)
            data.append(imagePath)
            orig = im.fromarray(img)
            orig.save(ouputPath + os.path.sep + 
                  imagePath.split(os.path.sep)[-1].split('.')[-2] + '.png')

        else:
            orig_resized = cv2.resize(img, (256, 256))
            orig = im.fromarray(orig_resized)
            orig.save(ouputPath_resized + os.path.sep + 
                  imagePath.split(os.path.sep)[-1].split('.')[-2] + '.png')
            continue
    else:
        orig_resized = cv2.resize(img, (256, 256))
        orig = im.fromarray(orig_resized)
        orig.save(ouputPath_resized + os.path.sep + 
              imagePath.split(os.path.sep)[-1].split('.')[-2] + '.png')
        continue
    
print("\n"+"-------Done!--------")
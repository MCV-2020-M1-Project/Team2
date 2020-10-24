#!/usr/bin/env python

""" MCV - M1:  Introduction to human and computer vision
    Week 2 - Museum painting retreival
    Team 2 - Lali Bibilashvili
             Víctor Casales
             Jaume Pregonas
             Lena Tolstoy

    text_removal.py: functions for t3 -> text bounding box detection
"""

""" Imports """
import cv2
import numpy as np  

""" Constants """

""" Global variables """

""" Classes """

""" Functions """


def findBox(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        cnt = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        area_imagen = img.shape[0] * img.shape[1]

        area_max = 0
        for c in cnt[0]:
            area = cv2.contourArea(c)
            if area > 1000 and ((area / area_imagen) * 100 < 25):
                x, y, w, h = cv2.boundingRect(c)
                if area > area_max:
                    area_max = area
                    xm, ym, wm, hm = x, y, w, h

        if area_max == 0:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sobel = sobely + sobelx
            sobel = cv2.convertScaleAbs(sobel)
            sobel = (255 - sobel)
            sobel = cv2.GaussianBlur(sobel, (3, 3), 0)
            cnt2 = cv2.findContours(sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            area_max = 0
            for c in cnt2[0]:
                x, y, w, h = cv2.boundingRect(c)
                area = h * w
                if area > 500 and ((area / area_imagen) * 100) < 20:
                    if area > area_max:
                        area_max = area
                        xm, ym, wm, hm = x, y, w, h
        return xm, ym, wm, hm

def saveMask(filename,img,x,y,w,h):
    width = img.shape[1]
    height = img.shape[0]
    box_img = np.zeros((height,width), np.uint8)
    box_img[y:y+h,x:x+w] = 255
    cv2.imwrite("bounding_masks/"+filename,box_img)


#TODO implement functions that return the bounding box
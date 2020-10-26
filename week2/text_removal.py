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
import pickle

""" Constants """

""" Global variables """

""" Classes """

""" Functions """
def findBox(img, mask):
    x1 = y1 = x2 = y2 = None
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)

        if mask is not None:
            laplacian = cv2.bitwise_and(laplacian, mask)

        cnt = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        area_imagen = img.shape[0] * img.shape[1]

        area_max = 0
        for c in cnt[0]:
            area = cv2.contourArea(c)
            if area > 1000 and ((area / area_imagen) * 100 < 25):
                x, y, w, h = cv2.boundingRect(c)
                if area > area_max:
                    area_max = area
                    x1, y1, wm, hm = x, y, w, h

        if area_max == 0:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sobel = sobely + sobelx
            sobel = cv2.convertScaleAbs(sobel)
            sobel = (255 - sobel)
            sobel = cv2.GaussianBlur(sobel, (3, 3), 0)

            if mask is not None:
                sobel = cv2.bitwise_and(sobel, mask)

            cnt2 = cv2.findContours(sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            area_max = 0
            for c in cnt2[0]:
                x, y, w, h = cv2.boundingRect(c)
                area = h * w
                if area > 500 and ((area / area_imagen) * 100) < 20:
                    if area > area_max:
                        area_max = area
                        x1, y1, wm, hm = x, y, w, h
        if x1 is not None:
            x2 = x1 + wm
            y2 = y1 + hm
            
        return [x1, y1, x2, y2]

def saveMask(filename,img,bbox,src):
    width = img.shape[1]
    height = img.shape[0]
    box_img = np.ones((height,width), np.uint8)
    box_img[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 255
    box_img = (255-box_img)
    cv2.imwrite(filename,box_img)

def evaluateIoU(src):
    result = []
    with open('.pickle_data/bboxes.pkl', 'rb') as f:
        boxes_found = pickle.load(f)
    with open(src+"/text_boxes.pkl","rb") as r:
        boxes = pickle.load(r)
    truth = []

    for b in boxes:
        truth.append([b[0][0][0],b[0][0][1],b[0][2][0],b[0][2][1]])


    for r in range(len(truth)):
        xA = max(boxes_found[r][1], truth[r][1])
        yA = max(boxes_found[r][0], truth[r][0])
        xB = min(boxes_found[r][3], truth[r][3])
        yB = min(boxes_found[r][2], truth[r][2])

        interArea = max( 0, xB - xA +1) *max(0, yB - yA +1)
        bboxAArea = (boxes_found[r][2] - boxes_found[r][0] + 1) * (boxes_found[r][3] - boxes_found[r][1] + 1)
        bboxBArea = (truth[r][2] - truth[r][0] + 1) * (truth[r][3] - truth[r][1] + 1)

        iou = interArea / float(bboxAArea + bboxBArea - interArea)
        result.append(iou)

    return result

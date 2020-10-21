#!/usr/bin/env python

""" MCV - M1:  Introduction to human and computer vision
    Week 2 - Museum painting retreival
    Team 2 - Lali Bibilashvili
             Víctor Casales
             Jaume Pregonas
             Lena Tolstoy

    functions.py: histogram functions called by t1
"""

""" Imports """
import cv2
import numpy as np
import sys

""" Constants """

""" Global variables """

""" Classes """

""" Functions """
""" ----- T1 - HISTOGRAMS ----- """
def transform_color(image, color_space):
    """
    Transforms the color space of an image
    """
    if color_space == "RGB":
        return image 

    conv_color = {
        "Gray": cv2.COLOR_RGB2GRAY,
        "CieLAB": cv2.COLOR_RGB2LAB,        
        "YCbCr": cv2.COLOR_RGB2YCrCb,
        "HSV": cv2.COLOR_RGB2HSV,
    }

    if color_space not in conv_color:
        print("Color sapce is not in the list!")

    return cv2.cvtColor(image, conv_color[color_space])

def compute_1D_histogram(img, mask):
    """
    Computes the gray-scale histogram
    """
    gray = transform_color(img, "Gray")
    hist= cv2.calcHist([gray], [0], mask, [256], [0, 256])  # compute the histogram
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)

    return hist

def compute_2D_histogram(img, csp, ch1, ch2, mask):
    """
    Computes 2D histogram
    """
    img = transform_color(img, csp)

    # Correspondence between color space channel and channel number: 
    # RGB -> b:0, g:1, r:2
    # CieLab -> l:0, a:1, b:2
    # YCrCb -> y:0, cr:1, cb:2
    # HSV -> h:0, s:1, v:2
    hist_a = cv2.calcHist([img], [ch1], mask, [256], [0,256])
    hist_b = cv2.calcHist([img], [ch2], mask, [256], [0,256])

    hist = np.concatenate((hist_a, hist_b))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)

    return hist

def compute_3D_histogram(img, csp, mask):
    """
    Computes 3D histogram
    """
    img = transform_color(img, csp)
    hist_a = cv2.calcHist([img], [0], mask, [256], [0,256]) # b (RGB), l (CieLAB), Y (YCrCb), H (HSV) 
    hist_b = cv2.calcHist([img], [1], mask, [256], [0,256]) # g (RGB), a (CieLAB), Cr (YCrCb), S (HSV) 
    hist_c = cv2.calcHist([img], [2], mask, [256], [0,256]) # r (RGB), b (CieLAB), Cb (YCrCb), V (HSV)

    hist = np.concatenate((hist_a, hist_b, hist_c))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)

    return hist

def compute_multi_histo(img, lvl, descriptor, csp, ch1, ch2, mask):
    step_h = img.shape[0]/lvl
    step_w = img.shape[1]/lvl

    hist = None
    for x in range(lvl):
        x1 = int(step_h*x)
        x2 = int(step_h*(x+1))
        for y in range(lvl):
            y1 = int(step_w*y)
            y2 = int(step_w*(y+1))

            tile = img[x1:x2, y1:y2]

            if descriptor == "1D_hist":
                if hist is None:
                    hist = compute_1D_histogram(tile, mask)
                else:
                    hist = np.concatenate((hist, compute_1D_histogram(tile, mask)))
            elif descriptor == "2D_hist":
                if hist is None:
                    hist = compute_2D_histogram(tile, csp, ch1, ch2, mask)
                else:
                    hist = np.concatenate((hist, compute_2D_histogram(tile, csp, ch1, ch2, mask)))
            elif descriptor == "3D_hist":
                if hist is None:
                    hist = compute_3D_histogram(tile, csp, mask)
                else: 
                    hist = np.concatenate((hist, compute_3D_histogram(tile, csp, mask)))
            else:
                sys.exit("Descriptor not supported")
    return hist
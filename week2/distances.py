#!/usr/bin/env python

""" MCV - M1:  Introduction to human and computer vision
    Week 2 - Museum painting retreival
    Team 2 - Lali Bibilashvili
             Víctor Casales
             Jaume Pregonas
             Lena Tolstoy

    distances.py: distances functions, code from extacted from Team 6 (TY!)
"""

""" Imports """
import random as rnd
import numpy as np
import cv2
import pickle as pkl
import sys


""" Constants """

""" Global variables """

""" Classes """

""" Functions """
def get_euclidean_distance(descriptor_a, descriptor_b):
    '''
    Gets descriptors as numpy arrays and returns Euclidean distance (pylint get off my back)
    '''
    dif = descriptor_a - descriptor_b
    return np.sqrt(np.sum(dif*dif))


def get_l1_distance(descriptor_a, descriptor_b):
    '''
    Gets descriptors as numpy arrays and returns L1 distance
    '''
    return np.sum(abs(descriptor_a - descriptor_b))


def get_x2_distance(descriptor_a, descriptor_b):
    '''
    Gets descriptors as numpy arrays and returns X2 distance
    '''
    dif = descriptor_a - descriptor_b

    num = dif*dif
    den = descriptor_a+descriptor_b
    return np.sum(np.divide(num, den, out=np.zeros_like(num), where=den != 0))


def get_chisq_distance(descriptor_a, descriptor_b):
    '''
    Gets descriptors as numpy arrays and returns X2 distance
    '''
    dif = descriptor_a - descriptor_b

    num = dif*dif
    den = descriptor_a
    return np.sum(np.divide(num, den, out=np.zeros_like(num), where=den != 0))


def get_hist_intersection(descriptor_a, descriptor_b):
    '''
    Gets descriptors as numpy arrays and returns histogram intersection
    '''
    return np.sum(np.minimum(descriptor_a, descriptor_b))


def get_hellinger_kernel(descriptor_a, descriptor_b):
    '''
    Gets descriptors as numpy arrays and returns hellinger kernel (whatever that is)
    '''
    if any(descriptor_a < 0) or any(descriptor_b < 0):
        print('All descriptor entries should be positive')
        return -1
    return np.sum(np.sqrt(descriptor_a*descriptor_b))


def get_correlation(a, b):  
    '''
    Correlation, implemented according to opencv documentation on histogram comparison
    '''
    dev_a = (a - np.mean(a))
    dev_b = (b - np.mean(b))

    return np.sum(dev_a*dev_b) / np.sqrt(np.sum(dev_a*dev_a)*np.sum(dev_b*dev_b))


def display_comparison(a, b):
    '''
    Displays an iamge with both descriptors (as histograms) alongside calculated measures

    This is kind of a silly thing, more showy than anything, but it might be useful when triying to
    decide which distance works best
    '''
    # image
    display_m_img = np.zeros((460, 828, 3), dtype=np.uint8)

    distances = get_all_measures(a, b)
    # measures
    text = ['Euclidean: ' + str(round(distances['eucl'], 2)),
            'X2: ' + str(round(distances['x2'], 2)),
            'L1: ' + str(round(distances['l1'], 2)), 
            'Hist intersection: ' + str(round(distances['h_inter'], 2)),
            'Hellinger Kernel: ' + str(round(distances['hell_ker'], 2)),
            'Correlation: ' + str(round(distances['corr'], 2)),
            'Chi square:' + str(round(distances['chisq'], 2))
        ]

    # Draw histograms
    ## Some position parameters
    hist_sq_size = (512, 200)

    x_offset = 20
    bt_y_hist_1 = 220
    bt_y_hist_2 = 440

    measure_text_pos = (552, 20)

    ## Draw first hist
    for k, v in enumerate(a):
        cv2.line(display_m_img, (int(hist_sq_size[0]*k/len(a)) + x_offset, bt_y_hist_1),
                                (int(hist_sq_size[0]*k/len(a)) + x_offset, bt_y_hist_1 - int(hist_sq_size[1]*v/max(a))),
                                (0, 255, 0)
                                )

    ## Draw second hist
    for k, v in enumerate(b):
        cv2.line(display_m_img, (int(hist_sq_size[0]*k/len(b)) + x_offset, bt_y_hist_2), 
                                (int(hist_sq_size[0]*k/len(b)) + x_offset, bt_y_hist_2 - int(hist_sq_size[1]*v/max(b))),
                                (0, 0, 255)
                                )

    ## Display text
    y = measure_text_pos[1]
    for t in text:
        cv2.putText(display_m_img, t, (measure_text_pos[0], y), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 0, 0))
        y += 15

    cv2.imshow('Display', display_m_img)
    cv2.waitKey(0)

    return

def get_distance(src, bbdd, measure):
    if measure == "eucl":
        dist = get_euclidean_distance(src, bbdd)
    elif measure == "l1": # this
        dist = get_l1_distance(src, bbdd)
    elif measure == "x2": # this
        dist = get_x2_distance(src, bbdd)
    elif measure == "h_inter":
        dist = get_hellinger_kernel(src, bbdd)
    elif measure == "corr":
        dist = get_correlation(src, bbdd)
    elif measure == "chisq":
        dist = get_chisq_distance(src, bbdd)
    else:
        sys.exit("Measure not supported")

    return dist

def get_top_k_similar(src_histo, db_histos, measure, k=3):
        distances = {}

        for fn in db_histos:
            key = fn.split('_')[1]
            distances[key] = abs(get_distance(src_histo, db_histos[fn], measure))

        if measure in ("corr", "h_inter"):
            reversed = True
        else: #euclidean, l1 or x2 
            reversed = False
        
        results = [int(key) for key in sorted(distances, key=distances.get, reverse=reversed)[:k]]
        return results

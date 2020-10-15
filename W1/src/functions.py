#!/usr/bin/env python3

""" MCV - M1:  Introduction to human and computer vision
    Week 1 - Content Based Image Retrieval
    Team 2 - Lali Bibilashvili
             Víctor Casales
             Jaume Pregonas
             Lena Tolstoy

    functions.py: TODO
"""
import itertools
from collections import OrderedDict

""" Imports """
import cv2
import numpy as np
import math

""" Constants """
# here you can define all the constants you might need

""" Global variables """
# here you can definde global variables -> IF NEEDED, try not to use them at all costs

""" Classes """
# if needed

""" Functions """
""" ----- T1 - DESCRIPTORS ----- """
def transform_color(image, color_space):
    """
    Computes image descriptors

    :param image: TODO
    :param color_space: TODO
    """
    conv_color = {
        "Gray": cv2.COLOR_RGB2GRAY,
        "CieLAB": cv2.COLOR_RGB2LAB,        
        "YCbCr": cv2.COLOR_RGB2YCrCb,
        "HSV": cv2.COLOR_RGB2HSV,
    }

    if color_space not in conv_color:
        print("Color sapce is not in the list!")

    return cv2.cvtColor(image, conv_color[color_space])

def compute_gray_histogram(key,img):
    """
    Computes the gray-scale normalized histogram
    :param img: 
    :return: 
    """
    hist = dict()
    gray = transform_color(img[key],"Gray")
    hist= cv2.calcHist([gray], [0], None, [256], [0, 256])  # compute the histogram
    hist = cv2.normalize(hist, hist)  # normalize the values of the histogram
    return hist

""" ----- T2 - SIMILARITY MEASURES ----- """
def check(h_dataset, h_query):
    """ 
    Checks the sizes of two histograms and matches them
        
    :param h_dataset: matrix containing the n dataset histogram data
    :param h_query: matrix containing the test image histogram data
    """
    im_dataset, vals_dataset = np.shape(h_dataset)
    size_shape = np.shape(np.shape(h_query))[0]
    

    # Case 1:
    if size_shape == 1:
        h_query = h_query[None, ...]
        nims, nvals_query = np.shape(h_query)

    # Case 2:
    if size_shape > 1:
        print("query histogram is more bigger")
    else:
        nims, nvals_query = np.shape(h_query)

    # Case 3:
    if nvals_query != vals_dataset:
        print("Histogram bins dimensions don't match")
    else:
        h_query = np.tile(h_query, (im_dataset, 1))

    return h_query

def euclidean(h_dataset, h_query):

   # h_query = check(h_dataset, h_query)
    distance = math.sqrt(sum((h_dataset - h_query)** 2))
    return distance

def l1_distance(h_dataset, h_query):

    h_query = check(h_dataset, h_query)
    distance = np.sum(np.absolute(h_dataset - h_query), axis=1)

    return distance

def x2_distance(h_dataset, h_query):

    h_query = check(h_dataset, h_query)
    distance = np.sum(np.power((h_dataset - h_query), 2)/ (h_dataset + h_query),axis=1,)

    return distance

def hist_intersection(h_dataset, h_query):

    h_query = check(h_dataset, h_query)
    distance = 1/(np.sum(np.minimum(h_dataset, h_query), axis=1))
    return distance

def hellinger(h_dataset, h_query):

    h_query = check(h_dataset, h_query)
    distance = 1/(np.sum(np.sqrt(h_dataset * h_query), axis=1))
    return distance

def kl_divergence(h_dataset, h_query):

    h_query = check(h_dataset, h_query)
    distance = np.sum(h_query * np.log((h_query) / (h_dataset)), axis=1,)
    return distance

def calculate_distances(h_dataset, h_query, mode):
    if mode == "euclidean":
        return euclidean(h_dataset, h_query)
    elif mode == "l1_distance":
        return l1_distance (h_dataset, h_query)
    elif mode == "x2_distance":
        return x2_distance(h_dataset, h_query)
    elif mode == "hist_intersection":
        return hist_intersection(h_dataset, h_query)
    elif mode == "hellinger":
        return hellinger(h_dataset, h_query)
    elif mode == "kl_divergence":
        return kl_divergence(h_dataset, h_query)
    else:
        raise Exception("Not function")

""" ----- T3 - IMAGE COMPARISION  ----- """
# these functions have to be called by T4a and T4b
def calculate_diferences(h_dataset,h_query,mode):
    distances = dict()
    for query in h_query.keys():
        distance = dict()
        for dataset in h_dataset.keys():

            dist = calculate_distances(h_dataset[dataset],h_query[query],mode)
            distance[dataset] = dist
        distances[query] = distance

    return distances


""" ----- T4 - IMAGE COMPARISION  ----- """
""" Top k """
def get_top_k(differences,top_k):
    query = []
    results = []

    for diff in differences.keys():
        query.append(diff)
        results.append(list(OrderedDict(sorted(differences[diff].items(), key=lambda t: t[1])).items())[:top_k])

    return query,results


""" MAP@k """
# these functions have to be called by T6b


""" ----- T5 - QS2  ----- """
""" Background removal """
# these functions have to be called by T6a

""" Descriptors """


""" ----- T6 - QS2 Evaluation  ----- """
""" Top k """

""" MAP@k """

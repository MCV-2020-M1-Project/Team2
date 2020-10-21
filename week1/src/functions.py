#!/usr/bin/env python

""" MCV - M1:  Introduction to human and computer vision
    Week 1 - Content Based Image Retrieval
    Team 2 - Lali Bibilashvili
             Víctor Casales
             Jaume Pregonas
             Lena Tolstoy

    functions.py: TODO
"""

""" Imports """
import cv2
import numpy as np
import math
import itertools
import pickle
import os
from collections import OrderedDict
from matplotlib import pyplot as plt
import ml_metrics as metrics
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib import colors 

""" Constants """

""" Global variables """

""" Classes """

""" Functions """
""" ----- T1 - DESCRIPTORS ----- """
def transform_color(image, color_space):
    """
    Computes image descriptors
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

def compute_1D_histogram(img, fn, plot, store, mask=None):
    """
    Computes the gray-scale histogram
    """

    gray = transform_color(img, "Gray")
    hist= cv2.calcHist([gray], [0], mask, [256], [0, 256])  # compute the histogram

    if plot or store:
        fig = plt.figure()
        plt.title("1D Grayscale histogram of "+fn)
        plt.xlabel("Bins")
        plt.ylabel("# of pixels")
        plt.plot(hist)
        plt.xlim([0,256])
        if plot:
            plt.show()
        if store:
            plt.savefig('../results/1D_hist/'+fn)
        plt.close(fig)

    return hist

def compute_2D_histogram(img, fn, plot, store, mask=None):
    """
    Computes H & S histogram
    """
    hsv = transform_color(img, "HSV")
    hist = cv2.calcHist([hsv], [0,1], mask, [180, 256], [0, 180, 0, 256])

    if plot or store:
        fig = plt.figure()
        plt.title("2D HSV histogram of "+fn)
        plt.xlabel("Saturation")
        plt.ylabel("Hue")
        plt.imshow(hist)
        if plot:
            plt.show()
        if store:
            plt.savefig('../results/2D_hist/'+fn)
        plt.close(fig)

    return hist

def compute_RGB_3D_histogram(img, fn, plot, store, mask=None):
    """
    Computes RGB histogram
    """
    plt.close('all')
    hist = dict()
    b, g, r = cv2.split(img)

    if plot or store:
        fig = plt.figure(figsize=(8,4))
        plt.title("3D RGB histogram of "+fn)
        plt.xlabel('Bins')
        plt.ylabel('# of pixels')
        plt.xlim([0,256])

    for x, c in zip([b,g,r], ["b", "g", "r"]):
        h = cv2.calcHist([x], [0], None, [256], [0,256])
        hist[c] = h

        if plot or store:
            plt.plot(h, color=c)

    if plot or store:
        if plot:
            plt.show()
        if store:
            plt.savefig('../results/3D_hist/'+fn)
        plt.close(fig)

    return hist

def task1(images, descriptor, plot, store, masks = None):
    """
    Computes all the histograms from the images of a folder

    :param images: src images
    :param descriptor: kind of descriptor we want to extract from the images
    :param plot: boolean that indicates if plots of the descriptors must be shown while on execution
    :param store: boolean that indicates if plots must be stored in a result folder on the parent directory (must be created beforehand)
    :return: histograms 
    """
    histograms = dict()
    for fn in images:
        img = images[fn]
        
        if masks is not None:
            mask = masks[fn]
        else:
            mask = None

        if descriptor == "1D_hist":
            hist = compute_1D_histogram(img, fn, plot, store, mask)
        elif descriptor == "2D_hist":
            hist = compute_2D_histogram(img, fn, plot, store, mask)
        elif descriptor == "3D_hist":
            hist = compute_RGB_3D_histogram(img, fn, plot, store, mask)
        else:
            print("Incorrect descriptor")
            return 
        histograms[fn] = hist

    return histograms

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
        print("query histogram is bigger")
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
    #h_query = check(h_dataset, h_query)
    distance = sum(np.absolute(h_dataset - h_query))

    return distance

def x2_distance(h_dataset, h_query):
    h_query = check(h_dataset, h_query)
    distance = np.sum(np.power((h_dataset - h_query), 2)/ (h_dataset + h_query),axis=1,)

    return distance

def hist_intersection(h_dataset, h_query):
    #h_query = check(h_dataset, h_query)
    distance = 1/(np.sum(np.minimum(h_dataset, h_query), axis=1))
    return distance

def hellinger(h_dataset, h_query):
   # h_query = check(h_dataset, h_query)
    distance = sum(np.sqrt(h_dataset * h_query))
    return distance

def kl_divergence(h_dataset, h_query):
    h_query = check(h_dataset, h_query)
    distance = np.sum(h_query * np.log((h_query) / (h_dataset)), axis=1,)
    return distance

def calculate_distances(h_dataset, h_query, mode):
    if mode == "euclidean":
        return euclidean(h_dataset, h_query)
    elif mode == "l1":
        return l1_distance (h_dataset, h_query)
    elif mode == "x2":
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
def calculate_diferences(h_dataset, h_query, mode):
    distances = dict()
    for query in h_query.keys():
        distance = dict()
        for dataset in h_dataset.keys():
            dist = calculate_distances(h_dataset[dataset],h_query[query],mode)
            distance[dataset] = dist
        distances[query] = distance

    return distances

def task3(images_bbdd, images, measure):
    """
    Computes the distance between two sets of images

    :param images_bbdd: images from the databes 
    :param images: images from qsd1
    :param measure: measure we want to use to compute the distance
    :return distances: returns all the distances between images where
    """
    bbdd_hist = task1(images_bbdd, "1D_hist", False, False)
    qsd1_hist = task1(images, "1D_hist", False, False)

    distances = calculate_diferences(bbdd_hist, qsd1_hist, measure)
    #TODO poder enseñar los resultados al usuario de alguna manera como task 1 -> si no da tiempo suda
    print("Not implemented how to show the results to the user")

    return distances


""" ----- T4 - IMAGE COMPARISION  ----- """
""" Top k """
def get_top_k(differences,top_k,reverse = False):
    query = []
    results = []

    for diff in differences.keys():
        difference = []
        difference.append(int(diff))
        query.append(difference)
        if not reverse:
            resultados = list(OrderedDict(sorted(differences[diff].items(), key=lambda t: t[1])).items())[:top_k]
        else:
            resultados = list(OrderedDict(sorted(differences[diff].items(), key=lambda t: t[1])).items())[-top_k:]
        result = []
        for res in resultados:
            result.append(int(res[0]))
        results.append(result)

    return query,results

def task4(images_bbdd, images, measure):
    """
    Computes the top k imges and the MAP@k and stores it on a pkl file

    :param images_bbdd: images from the databes 
    :param images: images from qsd1
    :param measure: measure we want to use to compute the distance
    """
    distances = task3(images_bbdd, images, measure)
    if measure == 'euclidean':
        query,top_results = get_top_k(distances,10) #get a top 10 results
        with open('result_euclidean_T.pkl', 'wb') as output:  # write the results in a file
            pickle.dump([query,top_results], output)
        with open("result_euclidean_T.pkl", "rb") as fp:  # load the results in a file
            [loaded_query,loaded_results] = pickle.load(fp)
        #print(loaded_query,loaded_results)
    elif measure == 'l1':
        query, top_results = get_top_k(distances, 10)  # get a top 10 results
        with open('result_l1_T', 'wb') as output:  # write the results in a file
            pickle.dump([query, top_results], output)
        with open("result_l1_T", "rb") as fp:  # load the results in a file
            [loaded_query, loaded_results] = pickle.load(fp)
    elif measure == 'hellinger':
        query, top_results = get_top_k(distances, 10,True)  # get a top 10 results
        with open('result_hellinger_T.pkl', 'wb') as output:  # write the results in a file
            pickle.dump([query, top_results], output)
        with open("result_hellinger_T.pkl", "rb") as fp:  # load the results in a file
            [loaded_query, loaded_results] = pickle.load(fp)
        print(loaded_query,loaded_results)
    else:
        raise Exception("Measure not supported for task 4")
    

""" MAP@k """
def compute_mapk(test_results,develop_results):
    result = metrics.mapk(test_results,develop_results,5)
    return result

""" ----- T5 - QS2  ----- """
def segment_paintings(images):
    masks = dict()

    low_color = np.array([0, 0, 70])
    high_color = np.array([50, 70, 2000])
    for fn in images:
        img = images[fn]

        blur = cv2.blur(img,(5,5))
        blur0=cv2.medianBlur(blur,5)
        blur1= cv2.GaussianBlur(blur0,(5,5),0)
        blur2= cv2.bilateralFilter(blur1,9,75,75)

        hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)

        res = cv2.inRange(hsv, low_color, high_color)
        #kernel = np.ones((11,11),np.uint8)
        #res = cv2.erode(res,kernel,iterations = 1)
        #res = cv2.dilate(res,kernel,iterations = 1)
        #cv2.imshow(fn, res)
        #cv2.waitKey()
        res = cv2.bitwise_not(res)
        #cv2.imshow(fn, res)
        #cv2.waitKey()
        #res2 = cv2.bitwise_and(img,img, mask = res)
        masks[fn] = res
        #cv2.imshow(fn, res2)
        #cv2.waitKey()
    return masks

def task5(images, descriptor): # tiene q devolver máscaras y descriptores
    """
    Extracts the background from the qsd2 images and computes the descriptors of the foreground

    :param images: images we want to extact the background to
    :param descriptor: kind of descriptor we want to extract from the images
    :returns masks of segmented images and their descriptors
    """

    """ Code to plot 3D HSV graph and 3D RGB graph
    for fn in images: 
        plt.close('All')
        img = images[fn]
        r,g,b = cv2.split (img)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
        norm = colors.Normalize(vmin=-1, vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()
        axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")
        plt.savefig('../results/'+fn+'_RGB.png')
        plt.close(fig)

        #HSV
        hsv_img = transform_color(img, "HSV")
        h, s, v = cv2.split(hsv_img)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        plt.savefig('../results/'+fn+'_HSV.png')
        plt.close(fig)
    """
    src_images = {k:v for k,v in images.items() if 'jpg' in k}
    masks = segment_paintings(src_images)
    histograms = task1(src_images, descriptor, False, False, masks)
    return masks, histograms

""" ----- T6 - QS2 Evaluation  ----- """ # tiene q devolver precision, recall, f1 y map
def task6(images, descriptor):
    src_masks = {k:v for k,v in images.items() if 'png' in k}
    src_images = {k:v for k,v in images.items() if 'jpg' in k}

    new_masks = segment_paintings(src_images)
    
    precision = dict()
    recall = dict()
    f1 = dict()
    tp = fp = falsen = 0
    for fn in new_masks:
        filename = os.path.splitext(fn)
        new_mask = new_masks[fn]
        src_mask = src_masks[filename[0]+'.png']
        src_mask = transform_color(src_mask, 'Gray')

        rows, cols = new_mask.shape

        for i in range(rows):
            for j in range(cols):
                new_px = new_mask[i,j]
                src_px = src_mask[i,j]
                if new_px == (src_px == 255):
                    tp += 1
                elif (new_px != src_px) and (new_px == 255):
                    fp +=1
                elif (new_px != src_px) and (new_px == 0):
                    falsen += 1

        precision[fn] = tp / (tp + fp)
        recall[fn] = tp / (tp + falsen)
        f1[fn] = 2*precision[fn]*recall[fn] / (precision[fn]+recall[fn])
    
    return precision, recall, f1

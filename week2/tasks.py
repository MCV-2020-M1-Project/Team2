#!/usr/bin/env python

""" MCV - M1:  Introduction to human and computer vision
    Week 2 - Museum painting retreival
    Team 2 - Lali Bibilashvili
             Víctor Casales
             Jaume Pregonas
             Lena Tolstoy

    tasks.py: task functions for w2
"""

""" Imports """
import cv2
import numpy as np  
from matplotlib import pyplot as plt
from pathlib import Path
import ml_metrics as metrics
import os
import sys
import csv
# Own files
import histograms as histos
import text_removal as txt_rm
import bckg_subs as bs
import distances as dists

""" Constants """

""" Global variables """

""" Classes """

""" Functions """
def task1(images, lvl, descriptor, csp, ch1, ch2, plot, store, masks=None):
    """
    Computes all the histograms from the images of a folder
    """
    histograms = dict()
    for fn in images:
        img = images[fn]
        
        if masks is not None:
            mask = masks[fn]
        else:
            mask = None
        
        histograms[fn] = histos.compute_multi_histo(img, lvl, descriptor, csp, ch1, ch2, mask)

        if plot or store:
            fig = plt.figure()
            title = descriptor
            path = "../results/test_results/t1/"+descriptor+"_lvl"+str(lvl)+"/"
            if descriptor == "2D_hist":
                title += ', '+csp+', '+str(ch1)+' & '+str(ch2)
                path += csp+str(ch1)+str(ch2)+"/"
            elif descriptor == "3D_hist":
                title += ', '+csp
                path += csp+"/"
            title += ', file: '+fn + ', lvl:'+str(lvl)

            plt.title(title)
            plt.xlabel("Bins")
            plt.ylabel("# of pixels")
            plt.plot(histograms[fn])

            if plot:
                plt.show()
            if store:
                Path(path).mkdir(parents=True, exist_ok=True)
                plt.savefig(path+fn)
            plt.close(fig)

    return histograms

def task2(images, bbdd, pkl_file, bckg_method, descriptor, lvl, csp, ch1, ch2, measure, plot, store):
    src_masks = dict()
    if bckg_method is not None:
        src_masks = bs.background_substraction_folder(images, bckg_method, csp)
    else:
        src_masks = None
    # get ground truth

    src_histos = task1(images, lvl, descriptor, csp, ch1, ch2, False, False, src_masks)
    bbdd_histos = task1(bbdd, lvl, descriptor, csp, ch1, ch2, False, False)
    
    topk = [None] * len(src_histos)
    for fn in src_histos:
        src_histo = src_histos[fn]
        topk[int(fn)] = dists.get_top_k_similar(src_histo, bbdd_histos, measure, 1)
    print(topk)
    print(pkl_file)

    map_k = metrics.kdd_mapk(pkl_file, topk, 1)

    print(map_k)

    if store:
        with open('../results/results_t2.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Descriptor: '+descriptor, 'Measure: '+ measure, 'k: '+ str(1),"Bckg_Method: "+bckg_method, "Lvl: "+str(lvl)])
            writer.writerow(['Map@k: '+str(map_k)])
            writer.writerow(['Actual','Predicted'])
            for i in range(len(pkl_file)):
                writer.writerow([str(pkl_file[i]), str(topk[i])])


def task3(images, plot, store):
    i = 0
    for fn in images:
        img = images[fn]

        x, y, w, h = txt_rm.findBox(img)
        txt_rm.saveMask(str(i)+".png",img,x,y,w,h)
        i += 1
        bb = [(10, 10), (30, 30)]   # TODO: call the functions on text_removal.py that returns the bounding box of the text area, 
                                    # right now the code is thinked in a way that [(x1, y1), (x2, y2)] 
                                    #   x1,y1 are the coordinates of the left-top (respectively) corner of the bb 
                                    #   x2,y2 are the coordinates of the right-bottom (respectively) corner of the bb
        if plot or store:
            to_show = cv2.rectangle(img, bb[0], bb[1], color=(0,255,0), thickness=2)
            if plot:
                cv2.imshow("t3_"+fn, to_show)
                cv2.waitKey(0)
                cv2.destroyWindow("t3_"+fn)
            if store:
                path = "../results/t3/"
                Path(path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(path+fn+".png", to_show)

def task6(src_images, bckg_method, csp, plot, store):
    bckg_masks = dict()
    for fn in src_images:
        img = src_images[fn]

        
        
        if plot or store:
            if plot:
                cv2.imshow("t6_bckg_"+fn)
                cv2.waitKey(0)
                cv2.destroyWindow("t6_bckg_"+fn)
            if store:
                path = "../results/t6/bckg/"+bckg_method+"/"
                if bckg_method in ("mcst", "mcck"):
                    path += csp+"/"
                Path(path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(path+fn+".png", bckg_mask)
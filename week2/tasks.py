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
import os
# Own files
import histograms as histos

""" Constants """

""" Global variables """

""" Classes """

""" Functions """
def task1(images, lvl, descriptor, csp, ch1, ch2, plot, store, masks=None):
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
        
        histograms[fn] = histos.compute_multi_histo(img, lvl, descriptor, csp, ch1, ch2, mask)

        if plot or store:
            fig = plt.figure()
            title = descriptor
            path = "../results/t1/"+descriptor+"_lvl"+str(lvl)+"/"
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

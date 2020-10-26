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
import pickle
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
    src_histos = task1(images, lvl, descriptor, csp, ch1, ch2, False, False, src_masks)
    bbdd_histos = task1(bbdd, lvl, descriptor, csp, ch1, ch2, False, False)
    
    topk = [None] * len(src_histos)
    for fn in src_histos:
        src_histo = src_histos[fn]
        topk[int(fn)] = dists.get_top_k_similar(src_histo, bbdd_histos, measure, 1)

    map_k = metrics.kdd_mapk(pkl_file, topk, 1)

    if store:
        with open('../results/results_t2.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Descriptor: '+descriptor, 'csp: '+csp, 'Measure: '+ measure, 'k: '+ str(1),"Bckg_Method: "+bckg_method, "Lvl: "+str(lvl)])
            writer.writerow(['Map@k: '+str(map_k)])
            #writer.writerow(['Actual','Predicted'])
            #for i in range(len(pkl_file)):
            #    writer.writerow([str(pkl_file[i]), str(topk[i])])


def task3(images, bbdd, pkl_file, lvl, measure, k, plot, store, mask=None):
    bboxes = []
    masks = dict()
    pkl_bb = [None] * len(images)

    for fn in images:
        img = images[fn]
        width = img.shape[1]
        height = img.shape[0]

        bbox = txt_rm.findBox(img, mask)
        bboxes.append(bbox)

        box_img = np.ones((height,width), np.uint8)
        box_img[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 255
        box_img = (255-box_img)
        masks[fn] = box_img
    
        pkl_bb[int(fn)] = [bbox]
    
    src_histos = task1(images, lvl, '3D_hist', "RGB", None, None, False, False, masks)
    bbdd_histos = task1(bbdd, lvl, '3D_hist', "RGB", None, None, False, False)

    topk = [None] * len(images)
    pkl_topk = [None] * len(images)
    for fn in src_histos:
        src_histo = src_histos[fn]
        topk[int(fn)] = dists.get_top_k_similar(src_histo, bbdd_histos, measure, k)
        pkl_topk[int(fn)] = [topk[int(fn)]]

    map_k = metrics.kdd_mapk(pkl_file, topk, k)

    # store pkl file
    with open('../results/QST1/result'+str(lvl)+str(k)+'.pkl', 'wb') as f:
        pickle.dump(pkl_topk, f)
    with open('../results/QST1/text_boxes'+str(lvl)+str(k)+'.pkl', 'wb') as f:
        pickle.dump(pkl_bb, f)

    with open('../results/results_t3.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Measure: '+ measure, 'k: '+ str(k), "Lvl: "+str(lvl)])
        writer.writerow(['Map@k: '+str(map_k)])
        #writer.writerow(['Actual','Predicted'])
        #for i in range(len(pkl_file)):
        #    writer.writerow([str(pkl_file[i]), str(topk[i])])

def task6(src_images, bbdd, pickle_file, measure, lvl, k, plot, store):
    bckg_masks = bs.detect_multiple_paintings(src_images, "canny", "RGB")
    bbdd_histos = task1(bbdd, lvl, "3D_hist", "RGB", None, None, False, False)

    topk = [None]*len(bckg_masks)
    bboxes = [None]*len(bckg_masks)
    map_k = 0
    idx = 0

    for fn in bckg_masks:
        image_bkg_mask = bckg_masks[fn]
        img = src_images[fn]
        width = img.shape[1]
        height = img.shape[0]

        img_topk = []
        img_bboxes = []
        for mask in image_bkg_mask:
            bbox = txt_rm.findBox(img, mask)
            if bbox[0] is not None:
                img_bboxes.append(bbox)

                box_img = np.ones((height,width), np.uint8)
                box_img[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 255
                box_img = (255-box_img)
                mask = cv2.bitwise_and(mask, box_img)

            path = '../results/masks'+str(lvl)+str(k)+'/'
            Path(path).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(mask, path)
            mask_histo = histos.compute_multi_histo(img, lvl, "3D_hist", "RGB", None, None, mask)

            mask_topk = dists.get_top_k_similar(mask_histo, bbdd_histos, measure, 10)
            img_topk.append(mask_topk)

            #compute MAP 
            if(len(pickle_file[int(fn)]) == 2):
                if pickle_file[int(fn)][0] in mask_topk or pickle_file[int(fn)][1] in mask_topk:
                    map_k += 1
            elif(len(pickle_file[int(fn)]) == 1):
                if pickle_file[int(fn)][0] in mask_topk:
                    map_k += 1
            else:
                print("Esto no dbería estar pasando")
            idx += 1

        topk[int(fn)] = img_topk
        bboxes[int(fn)] = img_bboxes

        # compute MAP@k
        map_k = float(map_k)/idx
    
    # store pkl file
    with open('../results/QST2/result'+str(lvl)+str(k)+'.pkl', 'wb') as f:
        pickle.dump(topk, f)
    with open('../results/QST2/text_boxes'+str(lvl)+str(k)+'.pkl', 'wb') as f:
        pickle.dump(bboxes, f)
    with open('../results/results_t6.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Measure: '+ measure, 'k: '+ str(k), "Lvl: "+str(lvl)])
        writer.writerow(['Map@k: '+str(map_k)])
        
    
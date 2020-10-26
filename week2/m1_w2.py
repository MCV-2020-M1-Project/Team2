#!/usr/bin/env python

""" MCV - M1:  Introduction to human and computer vision
    Week 2 - Museum painting retreival
    Team 2 - Lali Bibilashvili
             Victor Casales
             Jaume Pregonas
             Lena Tolstoy

    m1_w2.py: main program
"""

""" Imports """
import argparse
import os
import numpy as np
import cv2
import tasks as tasks
import sys
import pickle as pkl
#sys.path.append(os.getcwd()[:os.getcwd().index('week2')])

""" Constants """
DESCRIPTORS = ("1D_hist", "2D_hist", "3D_hist")
COLOR_SPACE = ("RGB", "HSV", "CieLAB", "YCbCr")
BCKG_METHODS = ("msc", "mcst", "mcck", "canny", "watershed")
MEASURES = ("eucl", "l1", "x2", "h_inter", "hell_ker", "corr", "chisq")

""" Global variables """

""" Classes """

""" Functions """
def build_arg_parser(ap):
    ap.add_argument("-t", "--task", required=True, type=int, dest="task", \
        help="number of the task to execute: 1-6")          
    ap.add_argument("-src", "--source", required=True, dest="src", \
        help="path to the folder with the images to analyse (qsd1, qsd2, qtd1 or qtd2)")
    ap.add_argument("-d", "--descriptor", required=False, dest="descriptor", \
        help="descriptor name, possible descriptors: " + str(DESCRIPTORS))
    ap.add_argument("-lvl", "--level", required=False, type=int, dest="level", \
        help="level of the multiresolution histograms, must be an integer")
    ap.add_argument("-c", "--csp", required=False, dest="csp", \
        help="color space, possible color spaces: " + str(COLOR_SPACE))
    ap.add_argument("-ch1", "--channel1", required=False, type=int, dest="ch1", \
        help="channel selected to compute the 2D hist -> 0, 1 or 2 (Respetively b,g,r in RGB; l,a,b in CieLAB; Y,Cr,Cb in YCrCb; H,S,V in HSV)")
    ap.add_argument("-ch2", "--channel2", required=False, type=int, dest="ch2", \
        help="channel selected to compute the 2D hist -> 0, 1 or 2 (Respetively b,g,r in RGB; l,a,b in CieLAB; Y,Cr,Cb in YCrCb; H,S,V in HSV)")
    ap.add_argument("-bm", "--bkcg_method", required=False, dest="bckg_method", \
        help="method name, possible methods: " + str(BCKG_METHODS))
    ap.add_argument("-m", "--measure", required=False, dest="measure", \
        help="measure name, possible measure: " + str(MEASURES))
    ap.add_argument("-bbdd", "--bbdd", required=False, dest="bbdd", \
        help="path to the folder which contains the bbdd images")
    ap.add_argument("-plot", "--plot", required=False, dest="plot",\
        help="allows plotting the results from the tasks")
    ap.add_argument("-store", "--store", required=False, dest="store",\
        help="stores the results from the tasks in the results folder (see documentation)")
    ap.add_argument("-k", "--k", required=False, dest="k", type=int, \
        help="K value for MAP@K")

def read_images(dict, ext, folder):
    for filename in sorted(os.listdir(folder)):
        if filename.find(ext) != -1:
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                dict[os.path.splitext(filename)[0]] = img
            else:
                print("Image "+filename+" couldn't be open")

def load_images_from_folder(folder):
    images = dict()
    masks = dict()
    
    if not os.path.isdir(folder):
        sys.exit('Src path doesn\'t exist')

    read_images(images, '.jpg', folder)
    read_images(masks, '.png', folder)
    
    if len(images) == 0:
        sys.exit('The folder: '+ folder + 'doesn\'t contain any images')
    
    if len(masks) == 0:
        print("No masks found in this folder")

    return images, masks


""" Main """
def main():
    ap = argparse.ArgumentParser()
    build_arg_parser(ap)
    args = ap.parse_args()

    images = load_images_from_folder(args.src) 

    plot = store = False
    if args.plot is not None:
        plot = args.plot
    if args.store is not None:
        store = args.store

    if args.task == 1: 
        if args.descriptor is None or args.descriptor not in DESCRIPTORS:
            ap.error('A correct descriptor must be provided for task 1, possible descriptors: ' + str(DESCRIPTORS))
        elif args.level is None or args.level < 1:
            ap.error('A valid histogram division level must be provided for task 1')
        elif (args.descriptor == "3D_hist" or args.descriptor == "2D_hist") and (args.csp is None or  args.csp not in COLOR_SPACE):
            ap.error('A correct color space must be provided for 2D and 3D histograms, possible color spaces: ' + str(COLOR_SPACE))
        else:
            if args.descriptor == "2D_hist":
                if args.ch1 < 0 or args.ch1 > 2:
                    ap.error('ch1 must be an integer between 0 and 2')
                elif args.ch2 < 0 or args.ch2 > 2:
                    ap.error('ch2 must be an integer between 0 and 2')
                elif args.ch1 == args.ch2:
                    ap.error('ch1 and ch2 can\'t be the same')
            tasks.task1(images[0], args.level, args.descriptor, args.csp, args.ch1, args.ch2, plot, store)
    
    elif args.task == 2:
        if args.descriptor is None or args.descriptor not in DESCRIPTORS:
            ap.error('A correct descriptor must be provided for task 1, possible descriptors: ' + str(DESCRIPTORS))
        elif args.level is None or args.level < 1:
            ap.error('A valid histogram division level must be provided for task 2')
        elif (args.descriptor == "3D_hist" or args.descriptor == "2D_hist") and (args.csp is None or  args.csp not in COLOR_SPACE):
            ap.error('A correct color space must be provided for 2D and 3D histograms, possible color spaces: ' + str(COLOR_SPACE))
        elif args.measure is None or args.measure is None:
            ap.error('A correct measure must be provided for task 2, possible measures: '+ str(MEASURES))
        elif args.bckg_method in ("mcst", "mcck") and (args.csp is None or args.csp not in ("RGB", "HSV")):
            ap.error('A correct color space must be provided for mcst and mcck bckg_methods, possible csp: RGB, HSV')
        elif args.bbdd is None:
            ap.error('A path to the database must be provided for task 2')
        else:
            if args.descriptor == "2D_hist":
                if args.ch1 < 0 or args.ch1 > 2:
                    ap.error('ch1 must be an integer between 0 and 2')
                elif args.ch2 < 0 or args.ch2 > 2:
                    ap.error('ch2 must be an integer between 0 and 2')
                elif args.ch1 == args.ch2:
                    ap.error('ch1 and ch2 can\'t be the same')
        bbdd = load_images_from_folder(args.bbdd) 

        #Read ground truth from .pkl
        actual = [] 
        with open(os.path.join(args.src,"gt_corresps.pkl"), 'rb') as gtfile:
            actual = pkl.load(gtfile)

        tasks.task2(images[0], bbdd[0], actual, args.bckg_method, args.descriptor, args.level, args.csp, args.ch1, args.ch2, args.measure, args.plot, args.store)
    elif args.task == 3:
        if args.level is None or args.level < 1:
            ap.error('A valid histogram division level must be provided for task 3')
        elif  args.k is None or args.k < 1:
            ap.error('A valid k value be provided for task 3')
        elif args.bbdd is None:
            ap.error('A path to the database must be provided for task 3')
        elif args.measure is None or args.measure is None:
            ap.error('A correct measure must be provided for task 3, possible measures: '+ str(MEASURES))
        else:
            bbdd = load_images_from_folder(args.bbdd)
            with open(os.path.join(args.src,"gt_corresps.pkl"), 'rb') as gtfile:
                actual = pkl.load(gtfile)

            tasks.task3(images[0], bbdd[0], actual, args.level, args.measure, args.k, plot, store)
    elif args.task == 6:
        if args.level is None or args.level < 1:
            ap.error('A valid histogram division level must be provided for task 6')
        elif  args.k is None or args.k < 1:
            ap.error('A valid k value be provided for task 6')
        elif args.bbdd is None:
            ap.error('A path to the database must be provided for task 6')
        elif args.measure is None or args.measure is None:
            ap.error('A correct measure must be provided for task 6, possible measures: '+ str(MEASURES))
        else:
            bbdd = load_images_from_folder(args.bbdd)
            with open(os.path.join(args.src,"gt_corresps.pkl"), 'rb') as gtfile:
                actual = pkl.load(gtfile)
            #with open('../results/QST2/result.pkl', 'rb') as f:
            #    results = pkl.load(f)
            #with open('../results/QST2/text_boxes.pkl', 'rb') as f:
            #    text_boxes = pkl.load(f)
            #print(results)
            #print(text_boxes)
            
            tasks.task6(images[0], bbdd[0], actual, args.measure, args.level, args.k, plot, store)
    else:
        print("Task not found")

if __name__ == "__main__":
    main()
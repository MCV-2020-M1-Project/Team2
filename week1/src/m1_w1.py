#!/usr/bin/env python

""" MCV - M1:  Introduction to human and computer vision
    Week 1 - Content Based Image Retrieval
    Team 2 - Lali Bibilashvili
             Victor Casales
             Jaume Pregonas
             Lena Tolstoy

    m1_w1.py: main program
"""

""" Imports """
import argparse
import os
import numpy as np
import sys
import cv2
sys.path.append(os.getcwd()[:os.getcwd().index('src')])
import src.functions as functions
from pandas import Series


""" Constants """
DESCRIPTORS = ("1D_hist", "2D_hist", "3D_hist")
#COLOR_SPACE = ("CieLAB", "YCbCr", "RGB")
MEASURES = ("euclidean", "l1", "x2", "hist_intersection", "hellinger", "kl_divergence")

""" Global variables """

""" Classes """

""" Functions """
def build_arg_parser(ap):                       # here you can add all the flags you want our script to execute
                                                # script execution example: python m1_w1.py -t 1 -src "path/to/files" -any_extra_flag
                                                #                           python m1_w1.py --task 1 --source "path/to/files" -d "descriptor_name"
    ap.add_argument("-t", "--task", required=True, dest="task", \
        help="number of the task to execute: 1-6")          
    ap.add_argument("-src", "--source", required=True, dest="src", \
        help="path to the folder with the images to analyse")
    ap.add_argument("-d", "--descriptor", required=False, dest="descriptor", \
        help="descriptor name, possible descriptors: " + str(DESCRIPTORS))
    #ap.add_argument("-c", "--color", required=False, dest="color", \
    #    help="color space, possible color spaces: " + str(COLOR_SPACE))
    ap.add_argument("-m", "--measure", required=False, dest="measure", \
        help="measure name, possible measures: " + str(MEASURES))
    ap.add_argument("-src2", "--source2", required=False, dest="src2", \
        help="path to the  bbdd for task 3")
    ap.add_argument("-plot", "--plot", required=False, dest="plot",\
        help="allows plotting the results from the tasks")
    ap.add_argument("-store", "--store", required=False, dest="store",\
        help="stores the results from the tasks in the results folder (see documentation)")

def load_images_from_folder(folder):
    images = dict()
    
    if not os.path.isdir(folder):
        sys.exit('Src path doesn\'t exist')

    for filename in os.listdir(folder):
        img = functions.cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images[filename] = img
        else:
            print("Image "+filename+" couldn't be open")
    
    if len(images) == 0:
        sys.exit('The folder: '+ folder + 'doesn\'t contain any images')

    return images


""" Main """
def main():
    ap = argparse.ArgumentParser()
    build_arg_parser(ap)
    args = ap.parse_args()

    images = load_images_from_folder(args.src) 

    if args.task == "1": #generates image descriptors (histograms)
        if args.descriptor is None or args.descriptor not in DESCRIPTORS:
            ap.error('A correct descriptor must be provided for task 1, possible descriptors: ' + str(DESCRIPTORS))
        #elif args.descriptor == "3D_hist" and (args.color is None or  args.color not in COLOR_SPACE):
        #    ap.error('A correct color space must be provided for 3D histograms, possible color spaces: ' + str(COLOR_SPACE))
        else:
            functions.task1(images, args.descriptor, False, True)

    elif args.task == "2":
        print("Nothing to show here, execute task 3")

    elif args.task == "3":
        if args.src2 is None:
            ap.error('A source path with the museum images must be provided in order to execute task 3')
        #elif args.descriptor is None or args.descriptor not in DESCRIPTORS:
        #    ap.error('A correct descriptor must be provided for task 3, possible descriptors: ' + str(DESCRIPTORS))
        elif args.measure is None or args.measure not in MEASURES:
            ap.error('A correct measure must be provided for task 3, possible measures: ' + str(MEASURES))
        else:
            images_bbdd = load_images_from_folder(args.src2)
            functions.task3(images_bbdd, images, args.measure)

    elif args.task == "4":
        if args.src2 is None:  
            ap.error('A source path with the museum images must be provided in order to execute task 3')
        elif args.measure is None or args.measure not in MEASURES:
            ap.error('A correct measure must be provided for task 3, possible measures: ' + str(MEASURES))
        else:
            images_bbdd = load_images_from_folder(args.src2)
            
            functions.task4(images_bbdd, images, args.measure)
            
    elif args.task == "5":
        if args.descriptor is None or args.descriptor not in DESCRIPTORS:
            ap.error('A correct descriptor must be provided for task 1, possible descriptors: ' + str(DESCRIPTORS))
        else:
            functions.task5(images, args.descriptor)
    elif args.task == "6":
        precision, recall, f1 = functions.task6(images, args.descriptor)
        avg_p = Series([precision.values()]).mean()
        avg_r = Series([recall.values()]).mean()
        avg_f1 = Series([f1.values()]).mean()
        print("precision -> "+avg_p+", recall -> "+avg_r+", f1 -> "+avg_f1)

    else:
        ap.error("Task must be a number between 1 and 6")

if __name__ == "__main__":
    main()
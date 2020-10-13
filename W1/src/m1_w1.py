#!/usr/bin/env python

""" MCV - M1:  Introduction to human and computer vision
    Week 1 - Content Based Image Retrieval
    Team 2 - Lali Bibilashvili
             Víctor Casales
             Jaume Pregonas
             Lena Tolstoy

    m1_w1.py: main program
"""

""" Imports """
import argparse
from functions import *

""" Constants """
DESCRIPTORS = ("some_descriptor", "another_descriptor") #TODO: add descriptors names
MEASURES = ("some_measure", "another_measure") #TODO: add measures names

""" Global variables """
# here you can definde global variables -> IF NEEDED, try not to use them at all costs

""" Classes """
# if needed

""" Functions """
def build_arg_parser(ap):                       # here you can add all the flags you want our script to execute
                                                # script execution example: python m1_w1.py -t 1 -src "path/to/files" -any_extra_flag
                                                #                           python m1_w1.py --task 1 --source "path/to/files" -d "descriptor_name"
    ap.add_argument("-t", "--task", required=True, dest="task", \
        help="number of the task to execute: 1-6")          
    ap.add_argument("-src", "--source", required=True, dest="src", \
        help="path to the files to analyse: single image or folder with multiple images")
    ap.add_argument("-d", "--descriptor", required=False, dest="descriptor", \
        help="descriptor name, possible descriptors: " + str(DESCRIPTORS))
    ap.add_argument("-m", "--measure", required=False, dest="measure", \
        help="measure name, possible measures: " + str(MEASURES))
    ap.add_argument("-src2", "--source2", required=False, dest="src2", \
        help="path to the museum images for task 3")


""" Main """
def main():
    ap = argparse.ArgumentParser()
    build_arg_parser(ap)
    args = ap.parse_args()

    # we also should check if the path (src) exists and has images on it and download images or whatever --> maybe create a common function on functions.py that reads images? lo vamos viendo

    if args.task == "1":
        if args.descriptor is None or args.descriptor not in DESCRIPTORS:
            ap.error('A correct descriptor must be provided for task 1, possible descriptors: ' + str(DESCRIPTORS))
        else:
            print("blbalbalbalab")
            # TODO: call the function on functions.py
    elif args.task == "2":
        if args.measure is None or args.measure not in MEASURES:
            ap.error('A correct measure must be provided for task 2, possible measures: ' + str(MEASURES))
        else:
            print("blbalbalbalab2")
            # TODO: call the function on functions.py
    elif args.task == "3":
        if args.src2 is None: # TODO: also check if the path exists or if other conditions must be met
            ap.error('A source path with the museum images must be provided in order to execute task 3')
        else:
            print("blbalbalbalab3")
            # TODO: call the function on functions.py
    elif args.task == "4":
        print("blbalbalbalab4")
        # TODO: check conditions and if everything is ok call the correspondent function
    elif args.task == "5":
        print("blbalbalbalab5")
        # TODO: check conditions and if everything is ok call the correspondent function
    elif args.task == "6":
        print("blbalbalbalab6")
        # TODO: check conditions and if everything is ok call the correspondent function
    else:
        ap.error("Task must be a number between 1 and 6")




if __name__ == "__main__":
    main()
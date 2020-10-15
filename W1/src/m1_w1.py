
#!/usr/bin/env python3

""" MCV - M1:  Introduction to human and computer vision
    Week 1 - Content Based Image Retrieval
    Team 2 - Lali Bibilashvili
             Víctor Casales
             Jaume Pregonas
             Lena Tolstoy

    m1_w1.py: main program
"""
import pickle

""" Imports """
import argparse
import os
import numpy as np
import sys
sys.path.append(os.getcwd()[:os.getcwd().index('src')])
import src.functions as functions

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
    histograms_bbdd = dict()
    histograms_qsw1 = dict()
    if args.task == "1": #convert color image to gray level
        if args.descriptor is None or args.descriptor not in DESCRIPTORS:
            ap.error('A correct descriptor must be provided for task 1, possible descriptors: ' + str(DESCRIPTORS))
        else:
            for img in images.keys():
                gray = functions.transform_color(images[img],"Gray")

    elif args.task == "2":
        if args.measure is None or args.measure not in MEASURES:
            ap.error('A correct measure must be provided for task 2, possible measures: ' + str(MEASURES))
        else:
            for img in images:
                #cv2.imshow('image', img)
                cv2.waitKey(0)
                # TODO: call the function on functions.py and show the output to the user
    elif args.task == "3":
        if args.src2 is None: # TODO: also check if the path exists or if other conditions must be met
            ap.error('A source path with the museum images must be provided in order to execute task 3')
        else:
            images2 = load_images_from_folder(args.src2)
                
            for img in images.keys(): #compute histogram of database
                histograms_bbdd[img] = functions.compute_gray_histogram(img,images)

            for img in images2.keys(): #compute the gray histograms of query set
                histograms_qsw1[img] = functions.compute_gray_histogram(img,images2)

    elif args.task == "4":
        if args.src2 is None:  # TODO: also check if the path exists or if other conditions must be met
            ap.error('A source path with the museum images must be provided in order to execute task 3')
        else:
            images2 = load_images_from_folder(args.src2)

            for img in images.keys(): #compute the gray_histograms of database
                histograms_bbdd[img] = functions.compute_gray_histogram(img,images)

            for img in images2.keys(): #compute histograms of query set
                histograms_qsw1[img] = functions.compute_gray_histogram(img,images2)
            """
                euclidean distance method
            """
            distances = functions.calculate_diferences(histograms_bbdd,histograms_qsw1,"euclidean") #compute the euclidean distance between each image of queryset and database
            query,top_results = functions.get_top_k(distances,10) #get a top 10 results
            with open('result.pkl', 'wb') as output:  # write the results in a file
                pickle.dump([query,top_results], output)
            with open("result.pkl", "rb") as fp:  # load the results in a file
                [loaded_query,loaded_results] = pickle.load(fp)
            #print(loaded_query,loaded_results)
            """
                l1 distance method
            """

            distances = functions.calculate_diferences(histograms_bbdd, histograms_qsw1,"l1_distance")  # compute the L1 distance between each image of queryset and database
            query, top_results = functions.get_top_k(distances, 10)  # get a top 10 results
            with open('result.pkl', 'wb') as output:  # write the results in a file
                pickle.dump([query, top_results], output)
            with open("result.pkl", "rb") as fp:  # load the results in a file
                [loaded_query, loaded_results] = pickle.load(fp)

            """
                hellinger method
            """
            distances = functions.calculate_diferences(histograms_bbdd, histograms_qsw1,"hellinger")  # compute the L1 distance between each image of queryset and database
            query, top_results = functions.get_top_k(distances, 10,True)  # get a top 10 results
            with open('result.pkl', 'wb') as output:  # write the results in a file
                pickle.dump([query, top_results], output)
            with open("result.pkl", "rb") as fp:  # load the results in a file
                [loaded_query, loaded_results] = pickle.load(fp)
            print(loaded_query,loaded_results)
    elif args.task == "5":
        # TODO: check conditions
        for img in images:
            cv2.imshow('image', img)
            cv2.waitKey(0)
            # TODO: call the function on functions.py and show the output to the user
    elif args.task == "6":
        # TODO: check conditions
        for img in images:
            cv2.imshow('image', img)
            cv2.waitKey(0)
            # TODO: call the function on functions.py and show the output to the user
    else:
        ap.error("Task must be a number between 1 and 6")

if __name__ == "__main__":
    main()
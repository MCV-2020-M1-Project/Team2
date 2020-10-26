#!/usr/bin/env python

""" MCV - M1:  Introduction to human and computer vision
    Week 2 - Museum painting retreival
    Team 2 - Lali Bibilashvili
             Víctor Casales
             Jaume Pregonas
             Lena Tolstoy

    bckg_subs.py: background substraction functions, code from extacted from Team 6 (TY!)
"""

""" Imports """
import cv2
import numpy as np  
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import sys
from pathlib import Path
import glob
# Own files
import histograms as histos

""" Constants """

""" Global variables """

""" Classes """

""" Functions """
def method_similar_channels_jc(image, thresh):

    """
    image - image as an array
    thresh - threshold as int

    """

    img = image.astype(float)

    # get image properties.
    h, w = np.shape(img)[:2]

    b_g = abs(img[:, :, 0] - img[:, :, 1])
    b_r = abs(img[:, :, 0] - img[:, :, 2])
    g_r = abs(img[:, :, 1] - img[:, :, 2])

    mask_matrix = np.uint8(b_g < thresh) * np.uint8(b_r < thresh) * np.uint8(g_r < thresh)
    mask_matrix *= np.uint8(img[:, :, 0] > 100) * np.uint8(img[:, :, 1] > 100) * np.uint8(img[:, :, 2] > 100) 


    mask_matrix = 1 - mask_matrix
    return mask_matrix.astype(np.uint8)


def method_similar_channels(img, thresh):
    """
    image - image as an array
    thresh - threshold as int
    return: mask 
    """

    # get image properties.
    h, w, bpp = np.shape(img)
    mask_matrix = np.empty(shape=(h, w), dtype='uint8')

    # iterate over the entire image.
    for py in range(0, h):
        for px in range(0, w):
            #print(img[py][px])
            blue = img[py][px][0]
            green = img[py][px][1]
            red = img[py][px][2]
            b_g = abs(int(blue) - int(green))
            b_r = abs(int(blue) - int(red))
            g_r = abs(int(green) - int(red))
            # and bigger than 100 to not be black
            if (b_g < thresh) \
                    and (b_r < thresh) \
                    and (g_r < thresh) \
                    and (blue > 100 and green > 100 and red > 100):
                # print('similar value')
                mask_matrix[py][px] = 0
            else:
                mask_matrix[py][px] = 255

    #cv2.imshow("mask", mask_matrix)
    #cv2.waitKey(0)
    #cv2.destroyWindow("mask")
    return mask_matrix


def method_colorspace_threshold(img, x_range, y_range, z_range, csp):
    """
    x = [bottom,top]
    y = [bottom,top]
    z = [bottom,top]

    bottom - top has value from 0-255

    colorspace = "RGB", "HSV", "CieLAB", "YCbCr"

    return: mask
    """

    img = histos.transform_color(img, csp)

    # mask color
    lower = np.array([x_range[0], y_range[0], z_range[0]])
    upper = np.array([x_range[1], y_range[1], z_range[1]])
    mask_matrix = cv2.inRange(img, lower, upper)

    return mask_matrix

""" KMEANS """
def make_histogram(cluster):
    """
    Count the number of pixels in each cluster
    :param: KMeans cluster
    :return: numpy histogram
    """
    numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    hist, _ = np.histogram(cluster.labels_, bins=numLabels)
    hist = hist.astype('float32')
    hist /= hist.sum()
    return hist


def make_bar(height, width, color):
    """
    Create an image of a given color
    :param: height of the image
    :param: width of the image
    :param: BGR pixel values of the color
    :return: tuple of bar, rgb values, and hsv values
    """
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv_bar[0][0]
    return bar, (red, green, blue), (hue, sat, val)


def sort_hsvs(hsv_list):
    """
    Sort the list of HSV values
    :param hsv_list: List of HSV tuples
    :return: List of indexes, sorted by hue, then saturation, then value
    """
    bars_with_indexes = []
    for index, hsv_val in enumerate(hsv_list):
        bars_with_indexes.append((index, hsv_val[0], hsv_val[1], hsv_val[2]))
    bars_with_indexes.sort(key=lambda elem: (elem[1], elem[2], elem[3]))
    return [item[0] for item in bars_with_indexes]


def get_most_common_color(image, k):
    # START HERE
    img = image
    height, width, _ = np.shape(img)

    # reshape the image to be a simple list of RGB pixels
    image = img.reshape((height * width, 3))

    # we'll pick the 5 most common colors
    num_clusters = k
    clusters = KMeans(n_clusters=num_clusters)
    clusters.fit(image)

    # count the dominant colors and put them in "buckets"
    histogram = make_histogram(clusters)

    # then sort them, most-common first
    combined = zip(histogram, clusters.cluster_centers_)
    combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # finally, we'll output a graphic showing the colors in order
    bars = []
    hsv_values = []
    for index, rows in enumerate(combined):
        bar, rgb, hsv = make_bar(100, 100, rows[1])
        #print(f'Bar {index + 1}')
        #print(f'  RGB values: {rgb}')
        #print(f'  HSV values: {hsv}')
        hsv_values.append(hsv)
        bars.append(bar)

    # sort the bars[] list so that we can show the colored boxes sorted
    # by their HSV values -- sort by hue, then saturation
    sorted_bar_indexes = sort_hsvs(hsv_values)
    sorted_bars = [bars[idx] for idx in sorted_bar_indexes]

    #cv2.imshow('Sorted by HSV values', np.hstack(sorted_bars))
    #cv2.imshow(f'{num_clusters} Most Common Colors', np.hstack(bars))
    #cv2.waitKey(0)
    bgr = (bars[0][0][0][0],bars[0][0][0][1],bars[0][0][0][2])

    return bgr,hsv_values[0]

def method_mostcommon_color_kmeans(img, k, thresh, csp):
    """
    methods uses kmeans to find most common colors on the photo, based on this information
    it's filtering that color considering it a background.

    k - provides number of buckets for kmeans algorithm
    thresh - provides number that creates the filter of colors close to the most common one
    colorspcae - allows to choose from different colorspaces bgr to hsv
    save - indicates whether you want to save masks or not
    generate measures - generates measures if set to True instead of a mask. If you want to generate measures
                        against ground truth as image provide name of the imagea without extension

    return: mask or measures( if generate_measures = True)
    """

    bgr, hsv = get_most_common_color(img, k)

    if csp == 'RGB':
        # mask color
        lower = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
        upper = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
        mask_matrix = cv2.inRange(img, lower, upper)
        mask_matrix = cv2.bitwise_not(mask_matrix, mask_matrix)

    if csp == 'HSV':
        hsv = histos.transform_color(img, "HSV")
        # mask color
        lower = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
        upper = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])
        mask_matrix = cv2.inRange(hsv, lower, upper)
        mask_matrix = cv2.bitwise_not(mask_matrix, mask_matrix)

    return mask_matrix


def method_watershed(img):
    """
        Return a binary mask that disregards background using watershed algorithm.
        Assumes that the background is close to the boundaries of the image and that the painting is smooth.
        Param: image (BGR)
        return: mask (binary image)
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    y_lim = img.shape[0]
    x_lim = img.shape[1]

    # mask is all zeroes except for background and painting markers
    mask = np.zeros_like(img[:, :, 0]).astype(np.int32)

    # Background pixels will be set to 1, this assumes position 5,5 is background
    mask[5, 5] = 1

    # pixels belonging to painting are set to 255, assuming the painting is always at the center of the image
    mask[int(y_lim / 2), int(x_lim / 2)] = 255
    mask[int(y_lim / 2) - 20, int(x_lim / 2)] = 255
    mask[int(y_lim / 2) + 20, int(x_lim / 2)] = 255
    mask[int(y_lim / 2), int(x_lim / 2) - 20] = 255
    mask[int(y_lim / 2), int(x_lim / 2) + 20] = 255
    mask[y_lim - int(y_lim * 0.3), int(x_lim / 2) + 20] = 255

    mask = cv2.watershed(img, mask)
    mask = (mask > 1)*255  # binarize (watershed did classify background as 1, non background as -1 and painting as 255)

    return mask

def count_white_pxs(contour):
    x1, y1, w, h = cv2.boundingRect(contour)
    img = np.zeros( [y1+h+50, x1+w+50, 1], dtype="uint8")
    img = cv2.drawContours(img, [contour], -1, (255), thickness=cv2.FILLED)
    return cv2.countNonZero(img)

def method_canny(img):
    """
    Calculate background limits regarding painting by detecting lines belonging to painting's frame.
    Assumes a smooth background.
    :param img:  image (BGR)
    :return: binary mask
    """
    sigma = 0.33
    image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    
    median = np.median(image_gray)
    lower = int(max(0, (1.0 - sigma) * median ))
    upper = int(min(255, (1.0 + sigma) * median))
    canny = cv2.Canny(blurred, lower, upper)

    kernel = np.ones((15,15), np.uint8)
    closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    cv2.destroyAllWindows()

    cnts, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    painting_cnt = []
    img_w, img_h, _ = img.shape

    for cnt in cnts:
        _, _, w, h = cv2.boundingRect(cnt)
        if (w > 400 and h > 200) or (h > 400 and w > 200):
            painting_cnt.append(cnt)
    
    if len(painting_cnt) == 0:
        mask = np.zeros([img_w, img_h, 1], dtype="uint8")
        mask.fill(255)
        return mask

    if len(painting_cnt) > 2:
        painting_cnt = sorted(painting_cnt, key = count_white_pxs, reverse = True)[:2]
    
    final_mask = np.zeros([img_w, img_h, 1], dtype="uint8")
    for cnt in painting_cnt:
        mask = np.zeros([img_w, img_h, 1], dtype="uint8")
        mask = cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)
        final_mask = cv2.bitwise_or(final_mask, mask)
    
    return final_mask


def background_substraction(img, method, csp):
    if method == "msc":
        mask = method_similar_channels(img, 30)
    elif method == "mcst":
        if csp == "RGB":
            mask = method_colorspace_threshold(img, [124, 255], [0, 255], [0, 255], "RGB")
        elif csp == "HSV":
            mask = method_colorspace_threshold(img, [0, 255], [0, 255], [140, 255], "HSV")
        else:
            sys.exit("Colorspace not supported")
    elif method == "mcck":
        if csp == "RGB":
            mask = method_mostcommon_color_kmeans(img, 5, 30, csp="RGB")
        elif csp == "HSV":
            mask = method_mostcommon_color_kmeans(img, 5, 10, csp="HSV")
        else:
            sys.exit("Colorspace not supported")
    elif method == "canny":
        mask = method_canny(img)
    elif method == "watershed":
        mask = method_watershed(img)
    else:
        sys.exit("Method not supported")

    return mask

def background_substraction_folder(images, method, csp):
    masks = dict()

    for fn in images:
        masks[fn] = background_substraction(images[fn], method, csp)

    return masks

def detect_multiple_paintings(images, method, csp):
    masks = dict()

    for fn in images:
        img = images[fn]
        initial_mask = background_substraction(img, method, csp)

        contours, _ = cv2.findContours(initial_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        image_masks = []
        painting_cnt = []
        img_w, img_h, _ = img.shape

        for cnt in contours:
            _, _, w, h = cv2.boundingRect(cnt)
            if (w > 400 and h > 200) or (h > 400 and w > 200):
                painting_cnt.append(cnt)

        if len(painting_cnt) == 0:
            mask = np.zeros([img_w, img_h, 1], dtype="uint8")
            mask.fill(255)
            image_masks.append(mask)

        if len(painting_cnt) > 2:
            painting_cnt = sorted(painting_cnt, key = count_white_pxs, reverse=True)[:2]
    
        for cnt in painting_cnt:
            mask = np.zeros([img_w, img_h, 1], dtype="uint8")
            mask = cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)
            image_masks.append(mask)
            
        masks[fn] = image_masks
        
        idx=1
        for mask in masks[fn]:
            path = "../results/t6/bckg/"+method+"/"
            if method in ("mcst", "mcck"):
                    path += csp+"/"
            Path(path).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(path+fn+"_"+str(idx)+".png", mask)
            idx += 1
    return masks
from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

from scipy.ndimage.filters import maximum_filter
from matplotlib import pyplot as plt
from itertools import chain
import sys, os, warnings
import random

from tqdm import tqdm
import numpy as np
import shutil
from PIL import Image
import cv2
import math

def patch_is_valid_care(patches, image_name, nb_patch):
    """ Check whether a patch contains too much noise, or not enough relevant information, which could bias the training
    Parameters
    ----------
        patch : a numpy array image
        image_name : name of the image
        nb_patch : number of patch treated
    Returns
    -------
        True if the patch is kept for training, False otherwise
    """
    # Calculate histogram of saturation channel
    patch, _ = patches
    patch = patch.astype('uint8')
    s = cv2.calcHist([patch], [1], None, [256], [0, 256])

    # Calculate attribute of the histogram
    pixel_values = np.arange(0, 256)
    mean_histo = np.sum(s.T * pixel_values) / np.sum(s)
    max_histo = np.max(s, axis=0)
    qty_high = np.sum(s[200:])

    if (mean_histo < 145 and max_histo > 1000 and qty_high < 210) or qty_high == 0.0 or mean_histo > 240:
        print("DELETED PATCH : " + image_name + '-' + str(nb_patch))
        cv2.imwrite('deleted_patches/patch_' + image_name + '_' + str(nb_patch) + '.png', patch)
        return False

    return True


def patch_is_valid_occupation_bis(patches, image_name, nb_patch, tshd_noise=100, tshd_occup=0.05):
    """ Check whether a patch contains too much noise, or not enough relevant information, which could bias the training
    Parameters
    ----------
        patch : a numpy array image
            Patch to be analyzed to keep or delete for training
        tshd_noise : int or float
            Threshold of pixel color above which contains relevant information (mitochondria)
        tshd_occup : int or float
            Required percentage of occupation of this relevant data
    Returns
    -------
        True if the patch is kept for training, False otherwise
    """
    # Print
    patch_x, patch_y = patches
    print("--------PATCH ", image_name, nb_patch, "-----------")

    # Calculate histogram of saturation channel
    patch_testing = patch_x.copy()
    s = cv2.calcHist([patch_testing], [0], None, [256], [0, 256])

    # Normalization of data
    patch_testing = patch_testing * 255.0 / patch_testing.max()

    # Savings histo for visu
    plt.figure(random.randint(0, 1000000))
    plt.plot(s)
    plt.savefig("histos/plot_" + image_name + '_' + str(nb_patch) + ".png")

    # Filtering of patches
    total_pixel = np.sum(s)
    occupation_min = tshd_occup * total_pixel
    patch_testing[patch_testing <= tshd_noise] = 0
    cv2.imwrite("todelete/patch_" + image_name + '_' + str(nb_patch) + "_bis.png", patch_testing)

    occupation = np.sum(np.where(patch_testing[:, :, 0] != 0, 1, 0))
    too_noisy = np.where(np.max(s[1:]) < 150, 1, 0) or np.where(occupation > total_pixel / 2, 1, 0)
    print("Pourcentage de blanc est pour patch :", image_name, '-', str(nb_patch), ":", occupation)

    # Delete if occupation is not enough
    if occupation < occupation_min:
        # Save deleted patch
        print("DELETED PATCH : less occup ")
        cv2.imwrite("deleted_patches/patch_" + image_name + '_' + str(nb_patch) + ".png", patch_y)
        return False

    if too_noisy:
        print("DELETED PATCH : too noisy ")
        cv2.imwrite("deleted_patches/patch_" + image_name + '_' + str(nb_patch) + ".png", patch_y)
        return False

    return True



def patch_is_valid_occupation(patches, image_name, nb_patch, occup_min):

    patch_y, patch_y_bool, patch_y_filter = patches
    total_pixel = patch_y.shape[0] * patch_y.shape[1]

    print(image_name, '-----', nb_patch)
    print(f"Occup min : {occup_min}")
    print(f"Occupation : {(np.sum(patch_y_bool) / total_pixel)}")

    if (np.sum(patch_y_bool) / total_pixel) < occup_min:
        print("DELETED PATCH ")
        cv2.imwrite("deleted_patches/" + image_name + '_' + str(nb_patch) + ".png", patch_y)
        return False

    cv2.imwrite("saved_filtered_patches/" + image_name + '_' + str(nb_patch) + ".png", patch_y_filter)

    return True
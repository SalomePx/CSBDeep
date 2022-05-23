from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageFilter
#from libtiff import TIFF
import numpy as np
import tqdm
import cv2
import sys
import os


def create_noised_inputs(data_path, gaussian_blur, gaussian_sigma, poisson_noise=False):
    """Create normalized training data to be used for neural network training.

    Parameters
    ----------
    data_path : str
        Path of folder where are saved data (training and testing)
    gaussian_blur : float: 0 if no gaussian filter, gaussian filter otherwise
        Provide gaussian filter to the image for low resolution dataset
    gaussian_sigma : int std
        Provide gaussian sigma noise (standard deviation) to the image for low resolution dataset
    poisson_noise : float: 0 if no gaussian filter, poisson noise otherwise
        Provide poisson noise to the image for low resolution dataset

    Returns
    -------
        Nothing
        Created input images for training and testing as low resolution images

    """

    folders = os.listdir(data_path)
    print(f"Creating perturbated images... ")

    for i, folder in enumerate(folders):

        files = os.listdir(data_path + "/" + folder + "/GT")
        print(f"Extracting folder {i+1}/{len(folders)}\n", end='\r')

        for j, file in enumerate(files):
            print(f"Adding noise to image {j + 1}/{len(files)}", end='\r')

            img_path = data_path + "/" + folder + "/GT/" + file
            img_noised_path = data_path + "/" + folder + "/low/" + file

            img = Image.open(img_path)
            img = np.array(img)
            img_noised = img.copy()

            # Apply noises
            if gaussian_filter:
                img_noised = gaussian_filter(img_noised, sigma=gaussian_blur)
            if gaussian_sigma is not None:
                noise = np.random.normal(0, gaussian_sigma, size=img_noised.shape).astype(np.float32)
                img_noised = np.maximum(0, img_noised + noise)
            if poisson_noise:
                img_noised = np.random.poisson(np.maximum(0, img_noised).astype(np.int)).astype(np.float32)

            cv2.imwrite(img_noised_path, img_noised)
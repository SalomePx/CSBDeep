from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter
import cv2


import numpy as np
import os


def create_noised_inputs(data_path, gaussian_blur, gaussian_noise, poisson_noise):
    """Create normalized training data to be used for neural network training.

    Parameters
    ----------
    data_path : str
        Path of folder where are saved data (training and testing)
    gaussian_filter : float: 0 if no gaussian filter, gaussian filter otherwise
        Provide gaussian filter to the image for low resolution dataset
    gaussian_noise : tuple (mean, var)
        Provide gaussian noise to the image for low resolution dataset
        If not, then tuple = None
    poisson_noise : float: 0 if no gaussian filter, poisson noise otherwise
        Provide poisson noise to the image for low resolution dataset

    Returns
    -------
        Nothing
        Created input images for training and testing as low resolution images

    """

    folders = os.listdir(data_path)
    for folder in folders: #train et test

        files = os.listdir(data_path + "/" + folder + "/GT")
        for file in files:
            img_path = data_path + "/" + folder + "/GT/" + file
            img_noised_path = data_path + "/" + folder + "/low/" + file
            img = Image.open(img_path)
            img_noised = np.array(img)

            # Apply noises
            if gaussian_filter:
                img_noised = gaussian_filter(img_noised, sigma=gaussian_blur)
                #img_noised = img_noised.filter(ImageFilter.GaussianBlur(radius = gaussian_filter))
            if gaussian_noise is not None:
                if img_noised.ndim == 3:
                    row, col, ch = img_noised.shape
                else:
                    row, col = img_noised.shape
                    ch = 1
                mean, var = gaussian_noise
                sigma = var ** 0.5
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                gauss = gauss.reshape(row, col, ch)
                if img_noised.ndim == 2:
                    gauss = gauss.squeeze()
                img_noised = img_noised + gauss
            if poisson_noise:
                vals = len(np.unique(img_noised))
                vals = 2 ** np.ceil(np.log2(vals))
                img_noised = np.random.poisson(img_noised * vals) / float(vals)

            cv2.imwrite(img_noised_path, img_noised)
            #img_noised.save(img_noised_path)
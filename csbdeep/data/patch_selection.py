from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import shutil
from PIL import Image
import cv2

def patch_is_valid_care(patches, image_name, nb_patch):
    """ Check whether a patch is relevant, contains enough info, or is only background
        Parameters
        ----------
        patches : a tuple of numpy array (input, ground_truth)
            Patch to be analyzed to keep or delete for training
        image_name : str
            Name of the image of the patch
        nb_patch :
            Number of the patch manipulated in the image
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


def patch_is_valid_occupation(patches, image_name, nb_patch, occup_min, verbose=False):
    """ Check whether a patch is relevant, contains enough info, or is only background
        Parameters
        ----------
            patches : a tuple of numpy array (input, ground_truth)
                Patch to be analyzed to keep or delete for training
            image_name : str
                Name of the image of the patch
            nb_patch :
                Number of the patch manipulated in the image
            occup_min : float between 0 and 1
                Minimum occupation for the relevant information. To adjust.
            verbose : bool
                Printing information of the function
        Returns
        -------
            True if the patch is kept for training, False otherwise
        """

    patch_y, patch_y_bool, patch_y_filter = patches
    total_pixel = patch_y.shape[0] * patch_y.shape[1]
    occupation = np.sum(patch_y_bool) / total_pixel

    if verbose:
        print(image_name, '-----', nb_patch)
        print(f"Occup min : {occup_min}")
        print(f"Occupation : {(np.sum(patch_y_bool) / total_pixel)}")

    if occupation < occup_min:
        if verbose:
            print("DELETED PATCH ")
        cv2.imwrite("savings/deleted_patches/" + image_name + '_' + str(nb_patch) + ".png", patch_y)
        return False

    return True
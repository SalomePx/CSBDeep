from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, backend_channels_last, normalize_0_255, save_figure, vrange
from ..utils.tf import keras_import
from csbdeep.data import no_background_patches_zscore

from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt
from scipy import signal, ndimage
from tifffile import imread
import tensorflow as tf
import numpy as np
import random
import math
import cv2
import sys

K = keras_import('backend')


def _mean_or_not(mean):
    # return (lambda x: K.mean(x,axis=(-1 if backend_channels_last() else 1))) if mean else (lambda x: x)
    # Keras also only averages over axis=-1, see https://github.com/keras-team/keras/blob/master/keras/losses.py
    return (lambda x: K.mean(x, axis=-1)) if mean else (lambda x: x)


def loss_laplace(mean=True):
    R = _mean_or_not(mean)
    C = np.log(2.0)
    if backend_channels_last():
        def nll(y_true, y_pred):
            n = K.shape(y_true)[-1]
            mu = y_pred[..., :n]
            sigma = y_pred[..., n:]
            return R(K.abs((mu - y_true) / sigma) + K.log(sigma) + C)

        return nll
    else:
        def nll(y_true, y_pred):
            n = K.shape(y_true)[1]
            mu = y_pred[:, :n, ...]
            sigma = y_pred[:, n:, ...]
            return R(K.abs((mu - y_true) / sigma) + K.log(sigma) + C)

        return nll


def loss_mae(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mae(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.abs(y_pred[..., :n] - y_true))

        return mae
    else:
        def mae(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.abs(y_pred[:, :n, ...] - y_true))

        return mae


def loss_mae_focus(mean=True):
    R = _mean_or_not(mean)

    def mae_focus(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        y_true_filter, y_pred_filter = filtered_loss(y_true_norm, y_pred_norm)
        return R(K.abs(y_pred_filter - y_true_filter))

    return mae_focus


def loss_mse(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mse(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.square(y_pred[..., :n] - y_true))

        return mse
    else:
        def mse(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.square(y_pred[:, :n, ...] - y_true))

        return mse


def loss_mse_focus(mean=True):
    R = _mean_or_not(mean)

    def mse_focus(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        y_true_filter, y_pred_filter = filtered_loss(y_true_norm, y_pred_norm)

        return R(K.square(y_pred_filter - y_true_filter))

    return mse_focus


def loss_psnre():
    def psnre(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        return tf.reduce_mean(tf.image.psnr(y_true_norm, y_pred_norm, 255))

    return psnre


def loss_psnr():
    def psnr(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        mask_true, mask_pred = loss_focus(y_true_norm, y_pred_norm)
        return - tf.reduce_mean(tf.image.psnr(mask_true, mask_pred, 255))

    return psnr


def loss_ssime():
    def ssime(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        return - tf.reduce_mean(tf.image.ssim(y_true_norm, y_pred_norm, 255))

    return ssime


def loss_ssim():
    def ssim(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        mask_true, mask_pred = loss_focus(y_true_norm, y_pred_norm)
        return - tf.reduce_mean(tf.image.ssim(mask_true, mask_pred, 255))

    return ssim


def loss_mae_ssim():
    def mae_ssim(y_true, y_pred):
        val_ssim = - loss_ssim_focus(y_true, y_pred)
        val_mae = - loss_mae_focus(y_true, y_pred)
        return -(0.3 * val_ssim + 0.7 * val_mae)

    return mae_ssim


def loss_mae_psnr():
    def mae_psnr(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        val_psnr = - loss_psnr_focus(y_true_norm, y_pred_norm)
        val_mae = - loss_mae_focus(y_true_norm, y_pred_norm)
        return -(0.3 * val_psnr + 0.7 * val_mae)

    return mae_psnr


def loss_psnr_ssim():
    def psnr_ssim(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        val_psnr = - loss_psnr_focus(y_true_norm, y_pred_norm)
        val_ssim = - loss_ssim_focus(y_true_norm, y_pred_norm)
        return -(0.3 * val_psnr + 0.7 * val_ssim)

    return psnr_ssim


def filtered_loss0(y_true, y_pred):

    batch_size = K.shape(y_true)[0]

    # Use numpy arrays
    array_true = y_true.numpy().squeeze()
    array_pred = y_pred.numpy().squeeze()

    # Calculate z-score and area of interest
    median_true = np.median(array_true, axis=(1, 2))
    std_true = np.std(array_true, axis=(1, 2))
    sub_true = np.array([[median_true, ] * array_true.shape[1], ] * array_true.shape[2]).T
    div_true = np.array([[std_true, ] * array_true.shape[1], ] * array_true.shape[2]).T
    zscore_true = (array_true - sub_true) / div_true
    zscore_true = np.where(zscore_true < 0, 0.0, zscore_true)
    mask_true = zscore_true > 1

    median_pred = np.median(array_pred, axis=(1, 2))
    std_pred = np.std(array_pred, axis=(1, 2))
    sub_pred = np.array([[median_pred, ] * array_pred.shape[1], ] * array_pred.shape[2]).T
    div_pred = np.array([[std_pred, ] * array_pred.shape[1], ] * array_pred.shape[2]).T
    zscore_pred = (array_pred - sub_pred) / div_pred
    zscore_pred = np.where(zscore_pred < 0, 0.0, zscore_pred)
    mask_pred = zscore_pred > 1

    common_mask = np.logical_or(mask_true, mask_pred)

    filtered_true = array_true[common_mask]
    filtered_pred = array_pred[common_mask]
    nb_nonzero = np.count_nonzero(common_mask, axis=(1, 2)).tolist()
    max_nonzero = max(nb_nonzero)

    q = max_nonzero // 11 + 1
    pixel_per_batch = q * 11

    final_true = np.zeros((batch_size, pixel_per_batch), dtype=object)
    final_pred = np.zeros((batch_size, pixel_per_batch), dtype=object)
    to_add_pixels = pixel_per_batch - np.array(nb_nonzero)

    # Fulfill with existing
    position_batch = np.repeat(np.arange(batch_size), np.array(nb_nonzero))
    position_1d = vrange(np.zeros((batch_size,), dtype=int), nb_nonzero)
    #position_1d = np.array([np.arange(k) for k in nb_nonzero])
    #position_1d = np.hstack(position_1d)

    final_true[position_batch, position_1d] = filtered_true
    final_pred[position_batch, position_1d] = filtered_pred

    # Fulfill with new
    rdm_pixel_true = [np.random.choice(array_true[i][common_mask[i]], to_add_pixels[i]) for i in range(batch_size)]
    rdm_pixel_true = np.hstack(rdm_pixel_true)
    rdm_pixel_pred = [np.random.choice(array_pred[i][common_mask[i]], to_add_pixels[i]) for i in range(batch_size)]
    rdm_pixel_pred = np.hstack(rdm_pixel_pred)

    position_batch_new = np.repeat(np.arange(batch_size), np.array(to_add_pixels))
    position_1d_new = vrange(nb_nonzero, to_add_pixels)
    #position_1d_new = np.array([np.arange(k, pixel_per_batch) for k in nb_nonzero])
    #position_1d_new = np.hstack(position_1d_new)

    final_true[position_batch_new, position_1d_new] = rdm_pixel_true
    final_pred[position_batch_new, position_1d_new] = rdm_pixel_pred

    final_true = np.reshape(final_true, [batch_size, q, 11, 1])
    final_pred = np.reshape(final_pred, [batch_size, q, 11, 1])

    final_true = np.asarray(final_true, dtype='float32')
    final_pred = np.asarray(final_pred, dtype='float32')

    return final_true, final_pred


def filtered_loss1(y_true, y_pred, threshold=0.2, perc=99.9):

    batch_size = K.shape(y_true)[0]
    img_height = K.shape(y_true)[1]
    img_width = K.shape(y_true)[2]
    nb_pixels = img_height * img_width

    array_true = y_true.numpy().squeeze()
    array_pred = y_pred.numpy().squeeze()
    masks_true = []
    masks_pred = []
    idx_common = []
    max_per_batch = 0
    nonzero_list = []

    for i in range(batch_size):
        # Calculate area of interest for true
        nb = str(random.randint(0, 10000))
        median_true = np.median(array_true[i])
        std_true = np.std(array_true)
        zscore_true = (array_true[i] - median_true) / std_true
        zscore_true = np.where(zscore_true < 0, 0.0, zscore_true)
        mask_true = zscore_true > 1

        # Calculate area of interest for pred
        median_pred = np.median(array_pred[i])
        std_pred = np.std(array_pred[i])
        zscore_pred = (array_pred[i] - median_pred) / std_pred
        zscore_pred = np.where(zscore_pred < 0, 0.0, zscore_pred)
        mask_pred = zscore_pred > 1

        # Visualize Z-score of patches
        common_mask = np.logical_or(mask_true, mask_pred)
        idx_common_mask = np.where(common_mask)
        filtered_true = np.where(common_mask, array_true[i], 0.0)
        filtered_pred = np.where(common_mask, array_pred[i], 0.0)
        #cv2.imwrite("todelete/pred_" + nb + ".tif", filtered_pred)
        #cv2.imwrite("todelete/true_" + nb + ".tif", filtered_true)

        # Calculate the vector with the max numer of interesting pixels
        nonzero_nb = np.count_nonzero(common_mask)

        nonzero_list.append(nonzero_nb)
        masks_true.append(filtered_true)
        masks_pred.append(filtered_pred)
        idx_common.append(idx_common_mask)

        if nonzero_nb > max_per_batch:
            max_per_batch = nonzero_nb

    q = max_per_batch // 11
    final_true = np.zeros((batch_size, q * 11))
    final_pred = np.zeros((batch_size, q * 11))
    to_add_list = max_per_batch - np.array(nonzero_list)
    to_add_real = q * 11 - np.array(nonzero_list)

    for i in range(batch_size):

        orig_true = masks_true[i]
        orig_true = orig_true[idx_common[i]]
        orig_pred = masks_pred[i]
        orig_pred = orig_pred[idx_common[i]]


        if to_add_list[i] > 0 and to_add_real[i] > 0:
            qty_rdm = to_add_real[i]
            loc_x = np.random.choice(np.arange(img_height), size=qty_rdm)
            loc_y = np.random.choice(np.arange(img_width), size=qty_rdm)

            # Locate random pixels - 1D array
            new_true = array_true[i, loc_x, loc_y]
            new_pred = array_pred[i, loc_x, loc_y]

            # Fulfilling final tensor
            final_true[i, :len(orig_true)] = orig_true
            final_true[i, len(orig_true):] = new_true
            final_pred[i, :len(orig_pred)] = orig_pred
            final_pred[i, len(orig_pred):] = new_pred

        elif to_add_list[i] == 0 or to_add_real[i] <= 0:
            final_true[i] = orig_true[:q*11]
            final_pred[i] = orig_pred[:q*11]

        else:
            raise Exception('The number should be positive')

    final_true = final_true.reshape((batch_size, q, 11, 1))
    final_pred = final_pred.reshape((batch_size, q, 11, 1))

    return final_true, final_pred


def filtered_loss2(y_true, y_pred):

    batch_size = K.shape(y_true)[0]
    batch_size = batch_size.numpy()
    nb_pixels = K.shape(y_true)[1] * K.shape(y_true)[2]

    array_true = y_true.numpy().squeeze()
    array_pred = y_true.numpy().squeeze()

    # Calculate area of interest for true
    median_true = np.median(array_true, axis=(1, 2))
    std_true = tf.math.reduce_std(y_true)
    filtered_true = (y_true - median_true) / std_true
    filtered_true = tf.where(filtered_true < 0, 0.0, filtered_true)
    mask_true = filtered_true > 1

    # Calculate area of interest for pred
    median_pred = np.median(array_pred)
    std_pred = tf.math.reduce_std(y_pred)
    filtered_pred = (y_pred - median_pred) / std_pred
    filtered_pred = tf.where(filtered_pred < 0, 0, filtered_pred)
    mask_pred = filtered_pred > 1

    # Calculate common area of interest
    common_mask = K.any(K.stack([mask_true, mask_pred], axis=0), axis=0)

    # Calculate new images and reshape in 1D
    new_y_true = tf.where(common_mask, y_true, 0.0)
    new_y_pred = tf.where(common_mask, y_pred, 0.0)
    new_y_true = tf.reshape(new_y_true, [batch_size, nb_pixels, 1, 1])
    new_y_pred = tf.reshape(new_y_pred, [batch_size, nb_pixels, 1, 1])

    # Delete zeros to not have a bias
    zero_vector = tf.zeros(shape=(batch_size, nb_pixels, 1, 1), dtype=tf.float32)
    bool_mask_true = tf.not_equal(new_y_true, zero_vector)
    bool_mask_pred = tf.not_equal(new_y_pred, zero_vector)

    focus_true = tf.boolean_mask(new_y_true, bool_mask_true)
    focus_pred = tf.boolean_mask(new_y_pred, bool_mask_pred)

    q = len(focus_true) // batch_size
    q2 = q // 11

    focus_true = tf.reshape(focus_true[:int(batch_size*q2*11)], [batch_size, q2, 11, 1])
    focus_pred = tf.reshape(focus_pred[:int(batch_size*q2*11)], [batch_size, q2, 11, 1])

    return focus_true, focus_pred


def filtered_loss_3(y_true, y_pred):

    batch_size = K.shape(y_true)[0]
    batch_size = batch_size.numpy()
    nb_pixels = K.shape(y_true)[1] * K.shape(y_true)[2]

    # Use numpy arrays
    array_true = y_true.numpy().squeeze()
    array_pred = y_pred.numpy().squeeze()

    # Calculate z-score and area of interest
    median_true = np.median(array_true, axis=(1, 2))
    std_true = np.std(array_true, axis=(1, 2))
    sub_true = np.array([[median_true, ] * array_true.shape[1], ] * array_true.shape[2]).T
    div_true = np.array([[std_true, ] * array_true.shape[1], ] * array_true.shape[2]).T
    zscore_true = (array_true - sub_true) / div_true
    zscore_true = np.where(zscore_true < 0, 0.0, zscore_true)
    mask_true = zscore_true > 1

    median_pred = np.median(array_pred, axis=(1, 2))
    std_pred = np.std(array_pred, axis=(1, 2))
    sub_pred = np.array([[median_pred, ] * array_pred.shape[1], ] * array_pred.shape[2]).T
    div_pred = np.array([[std_pred, ] * array_pred.shape[1], ] * array_pred.shape[2]).T
    zscore_pred = (array_pred - sub_pred) / div_pred
    zscore_pred = np.where(zscore_pred < 0, 0.0, zscore_pred)
    mask_pred = zscore_pred > 1

    common_mask = np.logical_or(mask_true, mask_pred)
    common_mask = np.expand_dims(common_mask, axis=3)
    print(common_mask.shape)
    common_mask = tf.convert_to_tensor(common_mask)

    filtered_true = tf.boolean_mask(y_true, common_mask)
    filtered_pred = tf.boolean_mask(y_pred, common_mask)

    nb_nonzero = np.count_nonzero(common_mask.numpy().squeeze(), axis=(1, 2)).tolist()
    max_nonzero = max(nb_nonzero)

    q = max_nonzero // 11 + 1
    pixel_per_batch = q * 11

    final_true = tf.zeros((batch_size, pixel_per_batch), dtype=object)
    final_pred = tf.zeros((batch_size, pixel_per_batch), dtype=object)
    to_add_pixels = pixel_per_batch - np.array(nb_nonzero)

    # Fulfill with existing
    position_batch = np.repeat(np.arange(batch_size), np.array(nb_nonzero))
    position_1d = np.array([np.arange(k) for k in nb_nonzero])
    position_1d = np.hstack(position_1d)

    tmp_true = np.zeros((len(position_1d), len(position_1d)))
    tmp_true[position_batch, position_1d] = 1
    tmp_true = tf.convert_to_tensor(tmp_true)

    # Fulfill with new
    rdm_pixel_true = [np.random.choice(array_true[i][common_mask[i]], to_add_pixels[i]) for i in range(batch_size)]
    rdm_pixel_true = np.hstack(rdm_pixel_true)
    rdm_pixel_pred = [np.random.choice(array_pred[i][common_mask[i]], to_add_pixels[i]) for i in range(batch_size)]
    rdm_pixel_pred = np.hstack(rdm_pixel_pred)

    position_batch_new = np.repeat(np.arange(batch_size), np.array(to_add_pixels))
    position_1d_new = np.array([np.arange(k, pixel_per_batch) for k in nb_nonzero])
    position_1d_new = np.hstack(position_1d_new)

    final_true[position_batch_new, position_1d_new] = rdm_pixel_true
    final_pred[position_batch_new, position_1d_new] = rdm_pixel_pred

    final_true = tf.reshape(final_true, [batch_size, q, 11, 1])
    final_pred = tf.reshape(final_pred, [batch_size, q, 11, 1])

    return final_true, final_pred




def filtered_loss4(y_true, y_pred):

    batch_size = K.shape(y_true)[0]
    batch_size = batch_size.numpy()
    img_height = K.shape(y_true)[1]
    img_width = K.shape(y_true)[2]
    nb_pixels = img_height * img_width

    # Use numpy arrays
    array_true = y_true.numpy().squeeze()
    array_pred = y_pred.numpy().squeeze()

    # Calculate z-score and area of interest
    median_true = np.median(array_true, axis=(1, 2))
    std_true = np.std(array_true, axis=(1, 2))
    sub_true = np.array([[median_true, ] * array_true.shape[1], ] * array_true.shape[2]).T
    div_true = np.array([[std_true, ] * array_true.shape[1], ] * array_true.shape[2]).T
    zscore_true = (array_true - sub_true) / div_true
    zscore_true = np.where(zscore_true < 0, 0.0, zscore_true)
    mask_true = zscore_true > 1

    median_pred = np.median(array_pred, axis=(1, 2))
    std_pred = np.std(array_pred, axis=(1, 2))
    sub_pred = np.array([[median_pred, ] * array_pred.shape[1], ] * array_pred.shape[2]).T
    div_pred = np.array([[std_pred, ] * array_pred.shape[1], ] * array_pred.shape[2]).T
    zscore_pred = (array_pred - sub_pred) / div_pred
    zscore_pred = np.where(zscore_pred < 0, 0.0, zscore_pred)
    mask_pred = zscore_pred > 1

    common_mask_arr = np.logical_or(mask_true, mask_pred)
    common_mask = np.expand_dims(common_mask_arr, axis=3)
    print(common_mask.shape)
    common_mask = tf.convert_to_tensor(common_mask)

    filtered_true = tf.boolean_mask(y_true, common_mask)
    filtered_pred = tf.boolean_mask(y_pred, common_mask)

    nb_nonzero = tf.math.count_nonzero(common_mask, axis=(1,2,3))

    rdm_pixel_true = [np.random.choice(array_true[i][common_mask_arr[i]], nb_pixels) for i in range(batch_size)]
    rdm_pixel_true = np.hstack(rdm_pixel_true)
    rdm_pixel_true = tf.convert_to_tensor(rdm_pixel_true)
    rdm_pixel_true = tf.reshape(rdm_pixel_true, [batch_size, img_height, img_width, 1])
    rdm_pixel_pred = [np.random.choice(array_pred[i][common_mask_arr[i]], nb_pixels) for i in range(batch_size)]
    rdm_pixel_pred = np.hstack(rdm_pixel_pred)
    rdm_pixel_pred = tf.convert_to_tensor(rdm_pixel_pred)
    rdm_pixel_pred = tf.reshape(rdm_pixel_pred, [batch_size, img_height, img_width, 1])

    final_true = tf.where(common_mask, y_true, rdm_pixel_true)
    final_pred = tf.where(common_mask, y_pred, rdm_pixel_pred)

    return final_true, final_pred

def filtered_loss_idx(y_true, y_pred):

    batch_size = K.shape(y_true)[0]
    batch_size = batch_size.numpy()
    img_height = K.shape(y_true)[1]
    img_width = K.shape(y_true)[2]
    nb_pixels = img_height * img_width

    # Use numpy arrays
    array_true = y_true.numpy().squeeze()
    array_pred = y_pred.numpy().squeeze()

    # Calculate z-score and area of interest
    median_true = np.median(array_true, axis=(1, 2))
    std_true = np.std(array_true, axis=(1, 2))
    sub_true = np.array([[median_true, ] * array_true.shape[1], ] * array_true.shape[2]).T
    div_true = np.array([[std_true, ] * array_true.shape[1], ] * array_true.shape[2]).T
    zscore_true = (array_true - sub_true) / div_true
    zscore_true = np.where(zscore_true < 0, 0.0, zscore_true)
    mask_true = zscore_true > 1

    median_pred = np.median(array_pred, axis=(1, 2))
    std_pred = np.std(array_pred, axis=(1, 2))
    sub_pred = np.array([[median_pred, ] * array_pred.shape[1], ] * array_pred.shape[2]).T
    div_pred = np.array([[std_pred, ] * array_pred.shape[1], ] * array_pred.shape[2]).T
    zscore_pred = (array_pred - sub_pred) / div_pred
    zscore_pred = np.where(zscore_pred < 0, 0.0, zscore_pred)
    mask_pred = zscore_pred > 1

    common_mask_arr = np.logical_or(mask_true, mask_pred)
    common_mask = np.expand_dims(common_mask_arr, axis=3)
    common_mask = tf.convert_to_tensor(common_mask)

    info = np.array(list(zip(*np.where(common_mask_arr))))
    idx_info = np.random.choice(np.arange(len(info)), nb_pixels).tolist()
    idx_mask = info[idx_info]
    print(idx_mask.shape)

    idx_filter = (idx_mask[:,0], idx_mask[:,1], idx_mask[:,2])
    filtered_true = array_true[idx_filter]
    print(img_height, img_width)
    filtered_true.reshape(img_height, img_width)
    filtered_true = tf.convert_to_tensor(filtered_true)
    filtered_true = tf.reshape(filtered_true, [batch_size, img_height, img_width, 1])

    filtered_pred = array_pred[idx_mask].reshape(img_height, img_width)
    filtered_pred = tf.convert_to_tensor(filtered_pred)
    filtered_pred = tf.reshape(filtered_pred, [batch_size, img_height, img_width, 1])

    '''
    rdm_pixel_true = [np.random.choice(array_true[i][common_mask_arr[i]], nb_pixels) for i in range(batch_size)]
    rdm_pixel_true = np.hstack(rdm_pixel_true)
    rdm_pixel_true = tf.convert_to_tensor(rdm_pixel_true)
    rdm_pixel_true = tf.reshape(rdm_pixel_true, [batch_size, img_height, img_width, 1])
    rdm_pixel_pred = [np.random.choice(array_pred[i][common_mask_arr[i]], nb_pixels) for i in range(batch_size)]
    rdm_pixel_pred = np.hstack(rdm_pixel_pred)
    rdm_pixel_pred = tf.convert_to_tensor(rdm_pixel_pred)
    rdm_pixel_pred = tf.reshape(rdm_pixel_pred, [batch_size, img_height, img_width, 1])
    '''

    final_true = tf.where(common_mask, y_true, filtered_true)
    final_pred = tf.where(common_mask, y_pred, filtered_pred)

    return final_true, final_pred


def loss_focus(y_true, y_pred):

    batch_size = K.shape(y_true)[0]
    batch_size = batch_size.numpy()
    img_height = K.shape(y_true)[1]
    img_width = K.shape(y_true)[2]
    nb_pixels = img_height * img_width

    array_true = y_true.numpy().squeeze()
    array_pred = y_pred.numpy().squeeze()

    # Calculate z-score and area of interest
    median_true = np.median(array_true, axis=(1, 2))
    std_true = np.std(array_true, axis=(1, 2))
    sub_true = np.array([[median_true, ] * array_true.shape[1], ] * array_true.shape[2]).T
    div_true = np.array([[std_true, ] * array_true.shape[1], ] * array_true.shape[2]).T
    zscore_true = (array_true - sub_true) / div_true
    zscore_true = np.where(zscore_true < 0, 0.0, zscore_true)
    mask_true = zscore_true > 1

    median_pred = np.median(array_pred, axis=(1, 2))
    std_pred = np.std(array_pred, axis=(1, 2))
    sub_pred = np.array([[median_pred, ] * array_pred.shape[1], ] * array_pred.shape[2]).T
    div_pred = np.array([[std_pred, ] * array_pred.shape[1], ] * array_pred.shape[2]).T
    zscore_pred = (array_pred - sub_pred) / div_pred
    zscore_pred = np.where(zscore_pred < 0, 0.0, zscore_pred)
    mask_pred = zscore_pred > 1

    common_mask_arr = np.logical_or(mask_true, mask_pred)
    common_mask = np.expand_dims(common_mask_arr, axis=3)
    common_mask_1d = np.reshape(common_mask, [batch_size, nb_pixels, 1, 1])
    common_mask = tf.convert_to_tensor(common_mask)

    # Reshape batches in 1D
    new_y_true = tf.where(common_mask, y_true, 0.0)
    new_y_pred = tf.where(common_mask, y_pred, 0.0)
    new_y_true = tf.reshape(new_y_true, [batch_size, nb_pixels, 1, 1])
    new_y_pred = tf.reshape(new_y_pred, [batch_size, nb_pixels, 1, 1])

    # Find localisation of zeros
    nb_nonzero = tf.math.count_nonzero(new_y_true,  axis=(1, 2, 3)).numpy()
    nb_zero = nb_pixels - nb_nonzero
    max_nonzero = int(np.max(nb_nonzero))
    mask_zeros = np.zeros((batch_size, nb_pixels, 1))

    loc_zeros = tf.where(common_mask_1d == False).numpy()
    loc_zeros_per_batch = [np.where(common_mask_1d[i]==0) for i in range(batch_size)]
    for i in range(batch_size):
        if i == 0:
            loc_zeros_per_batch.append(loc_zeros[:nb_zero[i]])
        elif i == batch_size-1:
            loc_zeros_per_batch.append(loc_zeros[nb_zero[i - 1]:nb_zero[i]])
        else:
            loc_zeros_per_batch.append(loc_zeros[:nb_zero[i - 1]])

    # Choose randomly zeros to delete
    print(loc_zeros_per_batch)
    idx_loc = [np.random.choice(np.arange(len(loc_zeros_per_batch[i])), replace=False) for i in range(batch_size)]
    for i, idx in zip(np.arange(batch_size), idx_loc):
        mask_zeros[i, idx] = 1
    mask_zeros = np.expand_dims(mask_zeros, axis=3)

    keep_true = tf.gather_nd(indices=common_mask_1d+mask_zeros, params=new_y_true)
    keep_pred = tf.gather_nd(indices=common_mask_1d+mask_zeros, params=new_y_pred)

    print(keep_pred)


    return y_true, y_pred



def loss_thresh_weighted_decay(loss_per_pixel, thresh, w1, w2, alpha):
    def _loss(y_true, y_pred):
        val = loss_per_pixel(y_true, y_pred)
        k1 = alpha * w1 + (1 - alpha)
        k2 = alpha * w2 + (1 - alpha)
        return K.mean(K.tf.where(K.tf.less_equal(y_true, thresh), k1 * val, k2 * val),
                      axis=(-1 if backend_channels_last() else 1))

    return _loss


def SNR(pred, target):
    """
    Compute Signal to Noise Ratio (SNR) of two images.
    Parameters
    ----------
    pred : array_like
      Prediction/comparison image
    target : array_like
      Reference image
    Returns
    -------
    x : float
      SNR of `pred` with respect to `target`
    """

    target_var = np.var(target)
    noise_var = mean_squared_error(target, pred)
    ratio = target_var / noise_var

    return 10.0 * np.log10(ratio)


def PSNR(pred, target):
    mse = np.mean((target - pred) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def PSNR_focus(pred, target, name_image='', save=False):
    img_filter = no_background_patches_zscore()
    pred_filter = img_filter(pred, image_name=name_image + '_pred', save=save)
    target_filter = img_filter(target, image_name=name_image + '_target', save=save)

    # Keep the union of information location
    common_filter = np.logical_or(pred_filter, target_filter)
    array_pred = pred[common_filter == 1]
    array_target = target[common_filter == 1]

    psnr = round(PSNR(array_pred, array_target), 3)
    return psnr


def SSIM_focus(pred, target, name_image='', save=False):
    img_filter = no_background_patches_zscore()
    pred_filter = img_filter(pred, image_name=name_image + '_pred', save=save)
    target_filter = img_filter(target, image_name=name_image + '_target', save=save)

    # Keep the union of information location
    common_filter = np.logical_or(pred_filter, target_filter)
    array_pred = pred[common_filter == 1]
    array_target = target[common_filter == 1]

    ssim = round(structural_similarity(array_pred, array_target, data_range=255), 3)
    return ssim


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def ssim_maps(y_pred, y_true, focus=False, perc=99.9, threshold=0.2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    y_pred = y_pred.astype(np.float32)
    y_true = y_true.astype(np.float32)
    y_pred, y_true = normalize_0_255([y_pred, y_true])

    if focus:
        # Calculate area of interest for true
        median_true = np.median(y_true)
        std_true = np.std(y_true)
        filtered_true = (y_true - median_true) / std_true
        filtered_true = np.where(filtered_true < 0, 0.0, filtered_true)
        perc_true = np.percentile(filtered_true.numpy(), perc)
        mask_true = filtered_true > threshold * perc_true

        # Calculate area of interest for pred
        median_pred = np.median(y_pred)
        std_pred = np.std(y_pred)
        filtered_pred = (y_pred - median_pred) / std_pred
        filtered_pred = np.where(filtered_pred < 0, 0, filtered_pred)
        perc_pred = np.percentile(filtered_pred, perc)
        mask_pred = filtered_pred > threshold * perc_pred

        # Calculate common area of interest
        common_mask = np.logical_or(mask_true, mask_pred)
        y_pred = np.where(common_mask == False, y_true, y_pred)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, y_pred, mode='valid')
    mu2 = signal.fftconvolve(window, y_true, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, y_pred * y_pred, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, y_true * y_true, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, y_pred * y_true, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))


def plot_multiple_ssim_maps(name_img, moment):
    x = imread('data_mito_crop/test/low/' + name_img + '.STED.ome.tif')
    x1 = imread('metric_test/bad/' + name_img + '.tif')
    x2 = imread('metric_test/mid/' + name_img + '.tif')
    x3 = imread('metric_test/good/' + name_img + '.tif')
    x4 = imread('metric_test/vgood/' + name_img + '.tif')
    y = imread('data_mito_crop/test/GT/' + name_img + '.STED.ome.tif')

    datas = [x, x1, x2, x3, x4, y]
    names = ['x', 'x1', 'x2', 'x3', 'x4', 'y']

    x, x1, x2, x3, x4, y = normalize_0_255(datas)
    psnrs = {}
    ssims = {}

    plt.figure(figsize=(16, 10))
    for i in range(len(datas)):
        key = names[i]
        if key == "y":
            ax = plt.subplot(2, len(datas), i + 1)
            ax.set_title('Target image')
            plt.imshow(y, cmap='gray')

            map = ssim_maps(datas[i], y)
            ax = plt.subplot(2, len(datas), i + 1 + len(datas))
            ax.set_title('Target image')
            plt.imshow(map, interpolation='nearest', cmap='viridis')

        else:
            psnrs[key], ssims[key] = PSNR_focus(datas[i], y, name_image=name_img + '_' + key, save=True), SSIM_focus(
                datas[i], y,
                name_img + '_' + key,
                save=True)
            print(f"PSNR {key} - target: {psnrs[key]} - SSIM: {ssims[key]}")
            ax = plt.subplot(2, len(datas), i + 1)
            ax.set_title(f'PSNR : {psnrs[key]} \n SSIM: {ssims[key]}')
            plt.imshow(datas[i], cmap='gray')

            map = ssim_maps(datas[i], y)
            ax = plt.subplot(2, len(datas), i + 1 + len(datas))
            plt.imshow(map, interpolation='nearest', cmap='viridis')

    plt.colorbar()
    plt.show()
    save_figure('ssim_maps', moment)

from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, backend_channels_last, normalize_0_255
from ..utils.tf import keras_import
from csbdeep.data import no_background_patches_zscore

import tensorflow as tf
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity

import tensorflow as tf
#import tensorflow_probability as tfp
import numpy as np
import cv2
import math
import random
import scipy.io
K = keras_import('backend')



def _mean_or_not(mean):
    # return (lambda x: K.mean(x,axis=(-1 if backend_channels_last() else 1))) if mean else (lambda x: x)
    # Keras also only averages over axis=-1, see https://github.com/keras-team/keras/blob/master/keras/losses.py
    return (lambda x: K.mean(x,axis=-1)) if mean else (lambda x: x)


def loss_laplace(mean=True):
    R = _mean_or_not(mean)
    C = np.log(2.0)
    if backend_channels_last():
        def nll(y_true, y_pred):
            n     = K.shape(y_true)[-1]
            mu    = y_pred[...,:n]
            sigma = y_pred[...,n:]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll
    else:
        def nll(y_true, y_pred):
            n     = K.shape(y_true)[1]
            mu    = y_pred[:,:n,...]
            sigma = y_pred[:,n:,...]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll


def loss_mae(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mae(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.abs(y_pred[...,:n] - y_true))
        return mae
    else:
        def mae(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.abs(y_pred[:,:n,...] - y_true))
        return mae


def loss_mse(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mse(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.square(y_pred[...,:n] - y_true))
        return mse
    else:
        def mse(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.square(y_pred[:,:n,...] - y_true))
        return mse


def loss_snr():
    def snr(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        return tf.reduce_mean(tf.image.psnr(y_true_norm, y_pred_norm, 255))
    return snr


def loss_ssim_global():
    def ssim_global(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        return - tf.reduce_mean(tf.image.ssim(y_true_norm, y_pred_norm, 255))
    return ssim_global


def loss_ssim():
    def ssim(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        mask_true, mask_pred = filtered_loss(y_true_norm, y_pred_norm)
        return - tf.reduce_mean(tf.image.ssim(mask_true, mask_pred, 255))

    return ssim


def filtered_loss(y_true, y_pred, threshold=0.2, perc=99.9):

    batch_size = K.shape(y_true)[0]
    nb_pixels = K.shape(y_true)[1] * K.shape(y_true)[2]

    array_true = y_true.numpy()
    array_pred = y_true.numpy()

    # Calculate area of interest for true
    median_true = np.median(array_true)
    std_true = tf.math.reduce_std(y_true)
    filtered_true = (y_true - median_true) / std_true
    filtered_true = tf.where(filtered_true < 0, 0.0, filtered_true)
    perc_true = np.percentile(filtered_true.numpy(), perc)
    mask_true = filtered_true > threshold * perc_true

    # Calculate area of interest for pred
    median_pred = np.median(array_pred)
    std_pred = tf.math.reduce_std(y_pred)
    filtered_pred = (y_pred - median_pred) / std_pred
    filtered_pred = tf.where(filtered_pred < 0, 0, filtered_pred)
    perc_pred = np.percentile(filtered_pred, perc)
    mask_pred = filtered_pred > threshold * perc_pred

    # Calculate common area of interest
    common_mask = K.any(K.stack([mask_true, mask_pred], axis=0), axis=0)

    # Calculate new images and reshape in 1D
    new_y_true = tf.where(common_mask, y_true, 0.0)
    new_y_pred = tf.where(common_mask, y_pred, 0.0)
    new_y_true = tf.reshape(new_y_true, [batch_size, nb_pixels, 1, 1])
    new_y_pred = tf.reshape(new_y_pred, [batch_size, nb_pixels, 1, 1])

    print("------NUMPY-----")
    print(new_y_true.numpy())

    # Delete zeros to not have a bias
    zero_vector = tf.zeros(shape=(batch_size, nb_pixels, 1, 1), dtype=tf.float32)
    bool_mask_true = tf.not_equal(new_y_true, zero_vector)
    bool_mask_pred = tf.not_equal(new_y_pred, zero_vector)

    focus_true = tf.boolean_mask(new_y_true, bool_mask_true)
    focus_pred = tf.boolean_mask(new_y_pred, bool_mask_pred)

    print("------FOCUS TRUE-----")
    print(focus_true.numpy())

    print(K.shape(bool_mask_true)[0])
    print(K.shape(bool_mask_true)[1])
    print(K.shape(bool_mask_true)[2])
    print(K.shape(bool_mask_true)[3])

    print("------NUMPY-----")
    print(focus_true.numpy())

    return focus_true, focus_pred


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

def PSNR_focus(pred, target, name_image=None, save=False):
    img_filter = no_background_patches_zscore()
    pred_filter = img_filter(pred, image_name=name_image+'_pred', save=save)
    target_filter = img_filter(target, image_name=name_image+'_target', save=save)

    # Keep the union of information location
    common_filter = np.logical_or(pred_filter, target_filter)
    array_pred = pred[common_filter==1]
    array_target = target[common_filter==1]

    psnr = round(PSNR(array_pred, array_target), 3)
    return psnr


def SSIM_focus(pred, target, name_image=None, save=False):
    img_filter = no_background_patches_zscore()
    pred_filter = img_filter(pred, image_name=name_image+'_pred', save=save)
    target_filter = img_filter(target, image_name=name_image+'_target', save=save)

    # Keep the union of information location
    common_filter = np.logical_or(pred_filter, target_filter)
    array_pred = pred[common_filter == 1]
    array_target = target[common_filter == 1]

    ssim = round(structural_similarity(array_pred, array_target, data_range=255),3)
    return ssim







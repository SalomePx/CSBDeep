from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, backend_channels_last
from ..utils.tf import keras_import

import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.image import ssim as k_ssim

import numpy as np
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

def loss_snr(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def snr(y_true, y_pred):
            n = K.shape(y_true)[-1]
            power_signal = K.var(y_true)
            divide = tf.cast(n, tf.float32)
            return R(power_signal / (K.square(y_true - y_pred[:,:n,...])) * divide)
        return snr
    else:
        def snr(y_true, y_pred):
            n = K.shape(y_true)[1]
            power_signal = K.var(y_true)
            divide = tf.cast(n, tf.float32)
            return R(power_signal / (K.square(y_true - y_pred[:,:n,...])) * (divide**2) )
        return snr

def loss_ssim(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def ssim(y_true, y_pred):
            n = K.shape(y_true)[-1]
            R(k_ssim(y_true, y_pred[...,:n], max_val=255))
        return ssim
    else:
        def ssim(y_true, y_pred):
            n = K.shape(y_true)[1]
            print(y_true)
            return R(k_ssim(y_true, y_pred[:,:n,...], max_val=255))
        return ssim


def loss_thresh_weighted_decay(loss_per_pixel, thresh, w1, w2, alpha):
    def _loss(y_true, y_pred):
        val = loss_per_pixel(y_true, y_pred)
        k1 = alpha * w1 + (1 - alpha)
        k2 = alpha * w2 + (1 - alpha)
        return K.mean(K.tf.where(K.tf.less_equal(y_true, thresh), k1 * val, k2 * val),
                      axis=(-1 if backend_channels_last() else 1))
    return _loss


def signal_to_noise(pred, target):
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




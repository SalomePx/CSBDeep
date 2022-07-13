from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, backend_channels_last, normalize_0_255, save_figure
from ..utils.tf import keras_import
from csbdeep.data import no_background_patches_zscore

from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt
from scipy import signal
from tifffile import imread
import tensorflow as tf
import numpy as np
import math

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
        y_true_filter, y_pred_filter = loss_focus(y_true, y_pred)
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
        y_true_filter, y_pred_filter = loss_focus(y_true, y_pred)
        return R(K.square(y_pred_filter - y_true_filter))

    return mse_focus


def loss_psnr():
    def psnr(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        return - tf.reduce_mean(tf.image.psnr(y_true_norm, y_pred_norm, 255))

    return psnr


def loss_psnr_focus():
    def psnr_focus(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        mask_true, mask_pred = loss_focus(y_true_norm, y_pred_norm)
        return - tf.reduce_mean(tf.image.psnr(mask_true, mask_pred, 255))

    return psnr_focus


def loss_ssim():
    def ssim(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        return - tf.reduce_mean(tf.image.ssim(y_true_norm, y_pred_norm, 255))

    return ssim


def loss_ssim_focus():
    def ssim_focus(y_true, y_pred):
        y_true_norm = y_true * 255.0 / tf.keras.backend.max(y_true)
        y_pred_norm = y_pred * 255.0 / tf.keras.backend.max(y_pred)
        mask_true, mask_pred = loss_focus(y_true_norm, y_pred_norm)
        return - tf.reduce_mean(tf.image.ssim(mask_true, mask_pred, 255))

    return ssim_focus


def loss_mae_ssim():
    def mae_ssim(y_true, y_pred):
        val_ssim = - loss_ssim_focus()(y_true, y_pred)
        val_mae = - loss_mae_focus()(y_true, y_pred)
        return -(0.5 * val_ssim + 0.5 * val_mae)

    return mae_ssim


def loss_mae_psnr():
    def mae_psnr(y_true, y_pred):
        val_psnr = - loss_psnr_focus()(y_true, y_pred)
        val_mae = - loss_mae_focus()(y_true, y_pred)
        return -(0.5 * val_psnr + 0.5 * val_mae)

    return mae_psnr


def loss_psnr_ssim():
    def psnr_ssim(y_true, y_pred):
        val_psnr = - loss_psnr_focus()(y_true, y_pred)
        val_ssim = - loss_ssim_focus()(y_true, y_pred)
        return -(0.5 * val_psnr + 0.5 * val_ssim)

    return psnr_ssim


def y_common_mask(y_true, y_pred):
    array_true = y_true.numpy().squeeze()
    array_pred = y_pred.numpy().squeeze()

    # Calculate z-score and area of interest
    mask_true = bool_interesting(array_true)
    mask_pred = bool_interesting(array_pred)

    common_mask = np.logical_or(mask_true, mask_pred)
    return common_mask


def bool_interesting(y):
    median_pred = np.median(y, axis=(1, 2))
    std_pred = np.std(y, axis=(1, 2))
    sub_pred = np.array([[median_pred, ] * y.shape[1], ] * y.shape[2]).T
    div_pred = np.array([[std_pred, ] * y.shape[1], ] * y.shape[2]).T
    zscore_pred = (y - sub_pred) / div_pred
    mask_pred = zscore_pred > 1
    return mask_pred


def loss_focus(y_true, y_pred):
    batch_size = K.shape(y_true)[0]
    batch_size = batch_size.numpy()
    img_height = K.shape(y_true)[1]
    img_width = K.shape(y_true)[2]
    nb_pixels = img_height * img_width
    array_true_1d = y_true.numpy().squeeze().reshape(batch_size, nb_pixels)
    array_pred_1d = y_pred.numpy().squeeze().reshape(batch_size, nb_pixels)

    # Common mask to focus on area of interest
    common_mask_arr = y_common_mask(y_true, y_pred)
    common_mask_arr = np.expand_dims(common_mask_arr, axis=3)
    common_mask_1d = np.reshape(common_mask_arr, [batch_size, nb_pixels])
    common_mask = tf.convert_to_tensor(common_mask_arr)

    # Reshape batches in 1D
    new_y_true = tf.where(common_mask, y_true, 0.0)
    new_y_pred = tf.where(common_mask, y_pred, 0.0)
    new_y_true = tf.reshape(new_y_true, [batch_size, nb_pixels, 1, 1])
    new_y_pred = tf.reshape(new_y_pred, [batch_size, nb_pixels, 1, 1])

    # Find localisation of zeros
    loc_nonzeros_per_batch = [np.where(common_mask_1d[i] == 1)[0] for i in range(batch_size)]
    loc_zeros_per_batch = [np.where(common_mask_1d[i] == 0)[0] for i in range(batch_size)]
    nb_zeros_per_batch = [len(loc_zeros_per_batch[i]) for i in range(batch_size)]
    nb_nonzeros_per_batch = [len(loc_nonzeros_per_batch[i]) for i in range(batch_size)]
    max_nonzeros = max(nb_nonzeros_per_batch)
    if max_nonzeros < 121:
        total_nonzero = 121
        q = max_nonzeros // 11
    else:
        q = max_nonzeros // 11 + 1
        total_nonzero = q * 11
    nb_zero_to_replace = total_nonzero - np.array(nb_nonzeros_per_batch)

    # Choose randomly zeros to save for each batch
    mask = np.where(common_mask_arr.reshape(batch_size, nb_pixels), 1, 0)
    idx_loc = [np.random.choice(np.arange(nb_zeros_per_batch[i]), size=nb_zero_to_replace[i], replace=False) for i in
               range(batch_size)]
    pixels_zero_to_change = [loc_zeros_per_batch[i][idx_loc[i]] for i in range(batch_size)]

    # Put to 2 zeros pixels to change with a random positive pixel
    for i, idx in zip(np.arange(batch_size), pixels_zero_to_change):
        mask[i, idx] = 2
    mask = np.reshape(mask, [batch_size, nb_pixels, 1, 1])
    mask = tf.convert_to_tensor(mask)

    # Select zeros indices to be replaced by random pixels
    # Position of zscore pixels
    pos_idx = [np.where(mask[i] == 1)[0] for i in range(batch_size)]
    # Number of positive pixels for each batch
    nb_idx = [np.arange(len(pos_idx[i])) for i in range(batch_size)]
    # Selection of random indices
    rdm_idx = [np.random.choice(nb_idx[i], size=nb_pixels) for i in range(batch_size)]
    rdm_idx_y = np.array([np.array(pos_idx[i][rdm_idx[i]]) for i in range(batch_size)]).reshape(
        batch_size * nb_pixels, )
    rdm_idx_x = np.repeat(np.arange(batch_size), nb_pixels)
    rdm_pixels_true = array_true_1d[(rdm_idx_x, rdm_idx_y)].reshape(batch_size, nb_pixels, 1, 1)
    rdm_pixels_pred = array_pred_1d[(rdm_idx_x, rdm_idx_y)].reshape(batch_size, nb_pixels, 1, 1)

    # Replace random pixels
    new_y_true = tf.where(mask == 2, rdm_pixels_true, new_y_true)
    new_y_pred = tf.where(mask == 2, rdm_pixels_pred, new_y_pred)
    new_y_true = tf.reshape(new_y_true, (batch_size, img_height, img_width, 1))
    new_y_pred = tf.reshape(new_y_pred, (batch_size, img_height, img_width, 1))

    # Select only values of the mask
    mask = tf.where(mask == 2, 1, mask)
    mask = tf.reshape(mask, (batch_size, img_height, img_width, 1))
    idx_mask = tf.where(mask == 1)
    keep_true = tf.gather_nd(indices=idx_mask, params=new_y_true)
    keep_pred = tf.gather_nd(indices=idx_mask, params=new_y_pred)
    keep_true = tf.reshape(keep_true, (batch_size, 11, q, 1))
    keep_pred = tf.reshape(keep_pred, (batch_size, 11, q, 1))

    return keep_true, keep_pred


def loss_focal(gamma=1, mean=True):
    R = _mean_or_not(mean)

    def focal(y_true, y_pred):
        y_true_arr = y_true.numpy().squeeze()
        y_pred_arr = y_pred.numpy().squeeze()
        maps = []
        for batch_true, batch_pred in zip(y_true_arr, y_pred_arr):
            p = ssim_maps(batch_true, batch_pred)
            p = p.astype('float32')
            maps.append(p)
        maps = tf.convert_to_tensor(maps)
        maps = tf.where(maps < 0, 0, maps)
        maps = tf.reshape(maps, y_true.shape)
        weighted_term = (1 - maps) ** gamma
        return R(tf.math.multiply(K.abs(y_pred - y_true), weighted_term))

    return focal


def loss_focal_mito(mean=True):
    R = _mean_or_not(mean)

    def focal_mito(y_true, y_pred):
        y_true_arr = y_true.numpy().squeeze()
        y_pred_arr = y_pred.numpy().squeeze()
        y_true_mask = bool_interesting(y_true_arr)

        gamma = tf.where(y_true_mask==True, 0.0, 1.0)
        gamma = tf.reshape(gamma, y_true.shape)
        maps = []

        for batch_true, batch_pred in zip(y_true_arr, y_pred_arr):
            p = ssim_maps(batch_true, batch_pred)
            p = p.astype('float32')
            maps.append(p)

        maps = tf.convert_to_tensor(maps)
        maps = tf.where(maps < 0, 0, maps)
        maps = tf.reshape(maps, y_true.shape)
        weighted_term = tf.math.pow((1 - maps), gamma)
        return R(tf.math.multiply(K.abs(y_pred - y_true), weighted_term))

    return focal_mito


def loss_focal_cristae(gamma=1, mean=True):
    R = _mean_or_not(mean)

    def focal_cristae(y_true, y_pred):
        p = - loss_ssim()(y_true, y_pred)
        weighted_term = (1 - p) ** gamma
        return R(K.abs(y_pred - y_true) * weighted_term)

    return focal_cristae


def loss_thresh_weighted_decay(loss_per_pixel, thresh, w1, w2, alpha):
    def _loss(y_true, y_pred):
        val = loss_per_pixel(y_true, y_pred)
        k1 = alpha * w1 + (1 - alpha)
        k2 = alpha * w2 + (1 - alpha)
        return K.mean(K.tf.where(K.tf.less_equal(y_true, thresh), k1 * val, k2 * val),
                      axis=(-1 if backend_channels_last() else 1))

    return _loss


def psnr(pred, target):
    mse = np.mean((target - pred) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def psnr_focus(target, pred, name_image='', save=False):
    img_filter = no_background_patches_zscore()
    pred_filter = img_filter(pred, image_name=name_image + '_pred', save=save)
    target_filter = img_filter(target, image_name=name_image + '_target', save=save)

    # Keep the union of information location
    common_filter = np.logical_or(pred_filter, target_filter)
    array_pred = pred[common_filter == 1]
    array_target = target[common_filter == 1]

    metric = round(psnr(array_pred, array_target), 3)
    return metric


def area_of_interest(y_true, y_pred, name_image='', save=False):
    img_filter = no_background_patches_zscore()
    pred_filter = img_filter(y_pred, image_name=name_image + '_pred', save=save)
    true_filter = img_filter(y_true, image_name=name_image + '_target', save=save)

    # Keep the union of information location
    common_filter = np.logical_or(pred_filter, true_filter)
    array_pred = y_pred[common_filter == 1]
    array_true = y_true[common_filter == 1]

    return array_true, array_pred


def ssim_focus(pred, target, name_image='', save=False):
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
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def ssim_maps(y_pred, y_true, focus=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)
    """

    y_pred = y_pred.astype(np.float32)
    y_true = y_true.astype(np.float32)
    y_pred, y_true = normalize_0_255([y_pred, y_true])

    if focus:
        # Calculate common area of interest
        common_mask = y_common_mask(y_true, y_pred)
        y_pred = np.where(common_mask == False, y_true, y_pred)

    size, sigma, L = 11, 1.5, 255
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    window = fspecial_gauss(size, sigma)
    mu1 = signal.fftconvolve(window, y_pred, mode='full')
    mu2 = signal.fftconvolve(window, y_true, mode='full')
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = signal.fftconvolve(window, y_pred * y_pred, mode='full') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, y_true * y_true, mode='full') - mu2_sq
    sigma12 = signal.fftconvolve(window, y_pred * y_true, mode='full') - mu1_mu2

    maps = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                        (sigma1_sq + sigma2_sq + C2))
    pad = int((size - 1) / 2)
    maps_pad = (maps[pad:maps.shape[0] - pad, pad:maps.shape[1] - pad])
    return maps_pad


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
            psnrs[key], ssims[key] = psnr_focus(y, datas[i], name_image=name_img + '_' + key, save=True), ssim_focus(
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

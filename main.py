from __future__ import print_function, unicode_literals, absolute_import, division

import os
import cv2
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
import datetime

from csbdeep.utils import download_and_extract_zip_file, plot_some, normalize, extract_zip_file
from csbdeep.data import RawData, create_patches, create_patches_mito, no_background_patches, norm_percentiles, sample_percentiles
from csbdeep.data.deteriorate import create_noised_inputs
from csbdeep.data.transform import flip_vertical, flip_90, flip_180, flip_270, zoom_aug
from csbdeep.data import no_background_patches_zscore

from csbdeep.utils import axes_dict, plot_some, plot_history, Path, download_and_extract_zip_file, save_figure, normalize_0_255
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.internals.losses import SNR, PSNR, PSNR_focus, SSIM_focus
from csbdeep.internals.predict import restore_and_eval_test, restore_and_eval
from skimage.metrics import structural_similarity


# -----------------------------
# -------- Time saving ----------
# -----------------------------
date = datetime.datetime.now()
month = date.month
day = date.day
hour = date.hour
min = date.minute
moment = str(month) + '-' + str(day) + '_' + str(hour) + '-' + str(min)


# ----------------------------------------------
# -------- Global parameters settings ----------
# ----------------------------------------------
# Fixed
initial_care = False
save_fig = True
load = False
testing = True
build_data = True
base_dir = 'fig/' + moment
tests = False

# Non fixed
create_patches_with_care = False
data_dir = 'data_mito_crop'

# -------------------------
# -------- Tests ----------
# -------------------------
if tests:
    x = imread('data_mito_crop/test/low/IMG0063.STED.ome.tif')
    x1 = imread('metric_test/bad/IMG0063.tif')
    x2 = imread('metric_test/mid/IMG0063.tif')
    x3 = imread('metric_test/good/IMG0063.tif')
    x4 = imread('metric_test/vgood/IMG0063.tif')
    y = imread('data_mito_crop/test/GT/IMG0063.STED.ome.tif')

    x, x1, x2, x3, x4, y = normalize_0_255([x, x1, x2, x3, x4, y])

    name_image = "IMG0063"
    psnr_x, ssim_x = PSNR_focus(x, y, name_image + '_init', save=True), SSIM_focus(x, y, name_image + '_init', save=True)
    psnr_x1, ssim_x1 = PSNR_focus(x1, y, name_image + '_x1', save=True), SSIM_focus(x1, y, name_image + '_x1', save=True)
    psnr_x2, ssim_x2 = PSNR_focus(x2, y, name_image + '_x2', save=True), SSIM_focus(x2, y, name_image + '_x2', save=True)
    psnr_x3, ssim_x3 = PSNR_focus(x3, y, name_image + '_x3', save=True), SSIM_focus(x3, y, name_image + '_x3', save=True)
    psnr_x4, ssim_x4 = PSNR_focus(x4, y, name_image + '_x4', save=True), SSIM_focus(x4, y, name_image + '_x4', save=True)

    print(f"PSNR init - target: {psnr_x} - SSIM: {ssim_x}")
    print(f"PSNR bad - target: {psnr_x1} - SSIM: {ssim_x1}")
    print(f"PSNR mid - target: {psnr_x2} - SSIM: {ssim_x2}")
    print(f"PSNR good - target: {psnr_x3} - SSIM: {ssim_x3}")
    print(f"PSNR very good - target: {psnr_x4} - SSIM: {ssim_x4}")

    plt.figure(figsize=(16, 10))
    ax = plt.subplot(1, 6, 1)
    ax.set_title(f'PSNR : {psnr_x} - SSIM: {ssim_x}', fontsize=8)
    plt.imshow(x, cmap='gray')
    ax1 = plt.subplot(1, 6, 2)
    ax1.set_title(f'PSNR : {psnr_x1} - SSIM: {ssim_x1}', fontsize=8)
    plt.imshow(x1, cmap='gray')
    ax2 = plt.subplot(1, 6, 3)
    ax2.set_title(f'PSNR : {psnr_x2} - SSIM: {ssim_x2}', fontsize=8)
    plt.imshow(x2, cmap='gray')
    ax3 = plt.subplot(1, 6, 4)
    ax3.set_title(f'PSNR : {psnr_x3} - SSIM: {ssim_x3}', fontsize=8)
    plt.imshow(x3, cmap='gray')
    ax4 = plt.subplot(1, 6, 5)
    ax4.set_title(f'PSNR : {psnr_x4} - SSIM: {ssim_x4}', fontsize=8)
    plt.imshow(x4, cmap='gray')
    ax5 = plt.subplot(1, 6, 6)
    ax5.set_title("Original image", fontsize=8)
    plt.imshow(y, cmap='gray')
    plt.savefig("todelete/metrics_" + moment + ".png")



# -----------------------------------------------------
# -------- Analysis with our patches method -----------
# -----------------------------------------------------
if not initial_care:

    if build_data:
        # Extract zip file
        extract_zip_file(
            folder_path = '/net/serpico-fs2/spapereu/' + data_dir,
            targetdir   = data_dir,
        )

        # Create noised images
        create_noised_inputs(
            data_path      = data_dir,
            gaussian_blur  = 3,
            gaussian_sigma = 5,
            poisson_noise  = False,
        )

    # Generate training data
    raw_data = RawData.from_folder(
        basepath    = data_dir + '/train',
        source_dirs = ['low'],
        target_dir  = 'GT',
        axes        = 'YX',
    )

    # Initialization of transforms
    axes = 'YX'
    flip_vertical = flip_vertical(axes)
    flip_90 = flip_90(axes)
    flip_180 = flip_180(axes)
    flip_270 = flip_270(axes)
    zoom_aug = zoom_aug(axes)

    if create_patches_with_care:
        X, Y, XY_axes = create_patches(
            raw_data            = raw_data,
            patch_size          = (128, 128),
            n_patches_per_image = 2,
            patch_filter        = no_background_patches(0),
            save_file           = data_dir + '/my_training_' + data_dir + '.npz',
        )

    else:
        # Creation of patches
        X, Y, XY_axes = create_patches_mito(
            raw_data            = raw_data,
            patch_size          = (128, 128),
            data_path           = data_dir,
            #transforms         = None,
            transforms          = [flip_vertical, flip_90, flip_180, flip_270],
            cut_or_sample_patch = 'sample',
            delete_black_patches = True,
            save_file           = data_dir + '/my_training_' + data_dir + '.npz',
        )

    # Split into training and validation data
    (X, Y), (X_val, Y_val), axes = load_training_data(data_dir + '/my_training_' + data_dir + '.npz', validation_split=0.1, verbose=True)
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

# ----------------------------------------------------
# -------- Analysis with CARE patch method -----------
# ----------------------------------------------------
else:
    # Download and extract zip file
    download_and_extract_zip_file(
        url       = 'http://csbdeep.bioimagecomputing.com/example_data/snr_7_binning_2.zip',
        targetdir = 'data',
        verbose   = 1,
    )

    # Generate training data
    raw_data = RawData.from_folder(
        basepath    = 'data/train',
        source_dirs = ['low'],
        target_dir  = 'GT',
        axes        = 'YX',
    )

    # Creation of patches
    X, Y, XY_axes = create_patches(
        raw_data            = raw_data,
        patch_size          = (128, 128),
        n_patches_per_image = 2,
        patch_filter        = no_background_patches(0),
        save_file           = 'data/my_training_data_bis.npz',
    )
    # Split into training and validation data
    (X, Y), (X_val, Y_val), axes = load_training_data('data/my_training_data_bis.npz', validation_split=0.2, verbose=True)
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


# ----------------------------------
# -------- Train a model -----------
# ----------------------------------

if not load:
    # CARE model
    config = Config(axes, n_channel_in, n_channel_out, train_loss='mae', unet_kern_size=3, train_batch_size=8, train_steps_per_epoch=88)
    vars(config)
    model = CARE(config, 'my_model', basedir=base_dir)
    model.keras_model.summary()
    history = model.train(X, Y, validation_data=(X_val, Y_val), epochs=10)

    # Plot history
    plt.figure(figsize=(16, 5))
    plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'], ['ssim', 'val_ssim'], ['snr', 'val_snr'])
    save_figure(moment, 'history')


# -----------------------------------
# -------- Load the model -----------
# -----------------------------------

model = CARE(config=None, name='my_model', basedir=base_dir, name_weights='best')


# ---------------------------------------------
# -------- Testings and predictions -----------
# ---------------------------------------------

if testing:
    psnrs, ssims, _, _ = restore_and_eval_test(model, 'YX', data_dir, moment)
    plt.figure()
    plt.hist(psnrs)
    save_figure(moment, "histo_psnr", 'PSNR', 'Qty')
    plt.figure()
    plt.hist(ssims)
    save_figure(moment, "histo_ssim", 'SSIM', 'Qty')


# -----------------------------------------------
# -------- Save plots and loss images -----------
# -----------------------------------------------
if save_fig:
    axes = 'YX'
    if initial_care:
        y = imread('data/test/GT/img_0001.tif')
        x = imread('data/test/low/img_0001.tif')
    elif 'care' in data_dir:
        y = imread('data_care/test/GT/img_0001.tif')
        x = imread('data_care/test/low/img_0001.tif')
    else:
        y = imread('data_mito_crop/train/GT/IMG0042.STED.ome.tif')
        x = imread('data_mito_crop/train/low/IMG0042.STED.ome.tif')

    # Predict a specific image
    restored = model.predict(x, axes)
    plt.figure(figsize=(15, 10))
    plot_some(np.stack([x, restored, y]), title_list=[['low', 'CARE', 'GT']], pmin=2, pmax=99.8)
    psnrs_plot, ssims_plot, psnrs_low_plot, ssims_low_plot = restore_and_eval((y, x, restored), original=True)
    plt.suptitle(f"Low: PSNR: {round(psnrs_low_plot[0], 2)} - SSIM: {round(ssims_low_plot[0], 2)}\n"
                 f"Prediction: PSNR: {round(psnrs_plot[0], 2)} - SSIM: {round(ssims_plot[0], 2)}")
    save_figure(moment, 'pred_img')

    # Watch signal per pixel
    plt.figure(figsize=(10, 5))
    for _x, _name in zip((x, restored, y), ('low', 'CARE', 'GT')):
        plt.plot(normalize(_x, 1, 99.7)[180], label=_name, lw=2)
    plt.legend()
    save_figure(moment, 'pixel_loss')

    # Evaluation
    plt.figure(figsize=(20, 12))
    #idx = [4,15,33,41,50]
    idx = [58,29,60,61,62]
    X_val_set = np.array([X_val[i] for i in idx])
    Y_val_set = np.array([Y_val[i] for i in idx])
    _P = model.keras_model.predict(X_val_set)
    psnrs, ssims = restore_and_eval((Y_val_set, X_val_set, _P))
    if config.probabilistic:
        _P = _P[..., :(_P.shape[-1]//2)]
    plot_some(X_val_set, Y_val_set, _P, pmax=99.5)
    plt.suptitle('5 example validation patches\n'      
                'top row: input (source),  '          
                'middle row: target (ground truth),  '
                'bottom row: predicted from source \n\n'
                 f'1 : PSNR: {round(psnrs[0], 2)} - SSIM: {round(ssims[0], 2)}             '
                 f'2 : PSNR: {round(psnrs[1], 2)} - SSIM: {round(ssims[1], 2)}             '
                 f'3 : PSNR: {round(psnrs[2], 2)} - SSIM: {round(ssims[2], 2)}             '
                 f'4 : PSNR: {round(psnrs[3], 2)} - SSIM: {round(ssims[3], 2)}             '
                 f'5 : PSNR: {round(psnrs[4], 2)} - SSIM: {round(ssims[4], 2)}')
    save_figure(moment, '5preds')

    # SAVE model
    model.export_TF()

# ---------------------------------
# -------- Load a model -----------
# ---------------------------------
if load:
    # Load images
    y = imread('data_care/test/GT/img_0001.tif')
    x = imread('data_care/test/low/img_0001.tif')
    axes = 'YX'

    # Load model
    model = CARE(config=None, name='my_model', basedir=base_dir, name_weights = 'weights_ninatubau.h5')
    restored = model.predict(x, axes)

    # Compare original prediction and ground truth
    plt.figure(figsize=(15, 10))
    plot_some(np.stack([x, restored, y]), title_list=[['low', 'CARE', 'GT']], pmin=2, pmax=99.8);
    save_figure(moment, '1pred')

    plt.figure(figsize=(10, 5))
    for _x, _name in zip((x, restored, y), ('low', 'CARE', 'GT')):
        plt.plot(normalize(_x, 1, 99.7)[180], label = _name, lw = 2)
    plt.legend()
    save_figure(moment, 'loss')
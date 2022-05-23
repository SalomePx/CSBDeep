from __future__ import print_function, unicode_literals, absolute_import, division
from operator import truediv
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import keras
import datetime
import os
import keras

from csbdeep.utils import download_and_extract_zip_file, plot_some, normalize, extract_zip_file
from csbdeep.data import RawData, create_patches, create_patches_mito, no_background_patches, norm_percentiles, sample_percentiles
from csbdeep.data.deteriorate import create_noised_inputs
from csbdeep.data.transform import flip_vertical, flip_90, flip_180, flip_270

from csbdeep.utils import axes_dict, plot_some, plot_history, Path, download_and_extract_zip_file, save_figure
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
from csbdeep.models import Config, CARE

# ----------------------------------------------
# -------- Global parameters settings ----------
# ----------------------------------------------
initial_care = False
save_fig = True
load = False
build_data = True
data_dir = 'data_mito_crop'

# -----------------------------
# -------- Keep time ----------
# -----------------------------
date = datetime.datetime.now()
month = date.month
day = date.day
hour = date.hour
min = date.minute
moment = str(month) + '-' + str(day) + '_' +  str(hour) + '-' + str(min)

# -----------------------------------------------------
# -------- Analysis with our patches method -----------
# -----------------------------------------------------
if not initial_care:

    if build_data:
        ### Extract zip file
        extract_zip_file (
            folder_path = '/net/serpico-fs2/spapereu/' + data_dir,
            targetdir   = data_dir,
        )

        ### Create noised images
        create_noised_inputs(
            data_path       = data_dir,
            gaussian_blur   = 3,
            gaussian_sigma  = 5,
            poisson_noise   = False,
        )

    ### Generate training data
    raw_data = RawData.from_folder (
        basepath    = data_dir + '/train',
        source_dirs = ['low'],
        target_dir  = 'GT',
        axes        = 'YX',
    )

    ### Creation of patches
    X, Y, XY_axes = create_patches_mito (
        raw_data            = raw_data,
        patch_size          = (128,128),
        data_path           = data_dir,
        #transforms         = [flip_vertical, flip_90, flip_180, flip_270] ,
        patch_filter        = no_background_patches(0),
        save_file           = data_dir + '/my_training_' + data_dir + '.npz',
    )

    ### Split into training and validation data
    (X, Y), (X_val, Y_val), axes = load_training_data(data_dir + '/my_training_' + data_dir + '.npz', validation_split=0.05, verbose=True)
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

# ----------------------------------------------------
# -------- Analysis with CARE patch method -----------
# ----------------------------------------------------
else:
    ### Download and extract zip file
    download_and_extract_zip_file(
        url       = 'http://csbdeep.bioimagecomputing.com/example_data/snr_7_binning_2.zip',
        targetdir = 'data',
        verbose   = 1,
    )

    ### Generate training data
    raw_data = RawData.from_folder(
        basepath    = 'data/train',
        source_dirs = ['low'],
        target_dir  = 'GT',
        axes        = 'YX',
    )

    ### Creation of patches
    X, Y, XY_axes = create_patches(
        raw_data            = raw_data,
        patch_size          = (128, 128),
        n_patches_per_image = 2,
        patch_filter        = no_background_patches(0),
        save_file           = 'data/my_training_data_bis.npz',
    )
    ### Split into training and validation data
    (X,Y), (X_val,Y_val), axes = load_training_data('data/my_training_data_bis.npz', validation_split=0.05, verbose=True)
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


# ----------------------------------
# -------- Train a model -----------
# ----------------------------------
if not load:
    ### CARE model
    config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=3, train_batch_size=8, train_steps_per_epoch=400)
    print(config)
    vars(config)
    model = CARE(config, 'my_model', basedir='models')
    model.keras_model.summary()
    history = model.train(X,Y, validation_data=(X_val,Y_val), epochs=25)

    ### Plot history
    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16,5))
    plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);
    save_figure(moment, 'history')

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
        y = imread('data_mito/test/GT/IMG0052.STED.ome.tif')
        x = imread('data_mito/test/low/IMG0052.STED.ome.tif')

    ### Load model
    model = CARE(config=None, name='my_model', basedir='models', name_weights='last')
    restored = model.predict(x, axes)

    ### Compare original prediction and ground truth
    plt.figure(figsize=(15, 10))
    plot_some(np.stack([x, restored, y]),
              title_list=[['low', 'CARE', 'GT']],
              pmin=2, pmax=99.8);
    save_figure(moment, '1pred')

    ### Watch signal per pixel
    plt.figure(figsize=(10, 5))
    for _x, _name in zip((x, restored, y), ('low', 'CARE', 'GT')):
        plt.plot(normalize(_x, 1, 99.7)[180], label=_name, lw=2)
    plt.legend()
    save_figure(moment, 'pixel_loss')

    ### Evaluation
    plt.figure(figsize=(20,12))
    _P = model.keras_model.predict(X_val[:5])
    if config.probabilistic:
        _P = _P[...,:(_P.shape[-1]//2)]
    plot_some(X_val[:5],Y_val[:5],_P,pmax=99.5)
    plt.suptitle('5 example validation patches\n'      
                'top row: input (source),  '          
                'middle row: target (ground truth),  '
                'bottom row: predicted from source');
    save_figure(moment, '5preds')

    ### SAVE model
    model.export_TF()

# ---------------------------------
# -------- Load a model -----------
# ---------------------------------
if load:
    ### Load images
    y = imread('data_care/test/GT/img_0001.tif')
    x = imread('data_care/test/low/img_0001.tif')
    axes = 'YX'

    ### Load model
    model = CARE(config=None, name='my_model', basedir='models', name_weights = 'weights_ninatubau.h5')
    restored = model.predict(x, axes)

    ### Compare original prediction and ground truth
    plt.figure(figsize=(15,10))
    plot_some(np.stack([x,restored,y]),
            title_list=[['low','CARE','GT']], 
            pmin=2,pmax=99.8);
    save_figure(moment, '1pred')

    plt.figure(figsize=(10,5))
    for _x,_name in zip((x,restored,y),('low','CARE','GT')):
        plt.plot(normalize(_x,1,99.7)[180], label = _name, lw = 2)
    plt.legend()
    save_figure(moment, 'loss')
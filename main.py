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

from csbdeep.utils import axes_dict, plot_some, plot_history, Path, download_and_extract_zip_file, save_figure
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
from csbdeep.models import Config, CARE

mito = False
care_noise = True
load = False

### Keep time
date = datetime.datetime.now()
month = date.month
day = date.day
hour = date.hour
min = date.minute
moment = str(month) + '-' + str(day) + '_' +  str(hour) + '-' + str(min)

if mito:
    ### Extract zip file
    extract_zip_file (
        folder_path = '/net/serpico-fs2/spapereu/data_mito',
        targetdir   = 'data_mito',
    )

    ### Create noised images
    create_noised_inputs(
        data_path       = 'data_mito',
        gaussian_blur   = 3,
        gaussian_noise  = (0,5),
        poisson_noise   = False,
    )

    ### Generate training data
    raw_data = RawData.from_folder (
        basepath    = 'data_mito/train',
        source_dirs = ['low'],
        target_dir  = 'GT',
        axes        = 'YX',
    )

    ### Creation of patches
    X, Y, XY_axes = create_patches_mito (
        raw_data            = raw_data,
        patch_size          = (128,128),
        patch_filter        = no_background_patches(0),
        save_file           = 'data/my_training_data_mito.npz',
    )

    ### Split into training and validation data
    (X, Y), (X_val, Y_val), axes = load_training_data('data/my_training_data_mito.npz', validation_split=0.05, verbose=True)
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

elif care_noise:
    ### Extract zip file
    extract_zip_file (
        folder_path = '/net/serpico-fs2/spapereu/data_care',
        targetdir   = 'data_care',
    )

    ### Create noised images
    create_noised_inputs(
        data_path       = 'data_care',
        gaussian_blur   = 0,
        gaussian_noise  = (0,5),
        poisson_noise   = False,
    )

    ### Generate training data
    raw_data = RawData.from_folder (
        basepath    = 'data_care/train',
        source_dirs = ['low'],
        target_dir  = 'GT',
        axes        = 'YX',
    )

    ### Creation of patches
    X, Y, XY_axes = create_patches_mito (
        raw_data            = raw_data,
        patch_size          = (128,128),
        patch_filter        = no_background_patches(0),
        save_file           = 'data/my_training_data_care.npz',
    )

    ### Split into training and validation data
    (X, Y), (X_val, Y_val), axes = load_training_data('data/my_training_data_care.npz', validation_split=0.05, verbose=True)
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


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

if not load:
    ### CARE model
    config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=3, train_batch_size=2, train_steps_per_epoch=400)
    print(config)
    vars(config)
    model = CARE(config, 'my_model', basedir='models')
    model.keras_model.summary()
    history = model.train(X,Y, validation_data=(X_val,Y_val), epochs=10)

    ### Plot history
    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16,5))
    plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);
    save_figure(moment, 'history')

if True:
    y = imread('data_care/test/GT/img_0001.tif')
    x = imread('data_care/test/low/img_0002.tif')
    axes = 'YX'

    ### Load model
    model = CARE(config=None, name='my_model', basedir='models', name_weights='last')
    restored = model.predict(x, axes)

    ### Compare original prediction and ground truth
    plt.figure(figsize=(15, 10))
    plot_some(np.stack([x, restored, y]),
              title_list=[['low', 'CARE', 'GT']],
              pmin=2, pmax=99.8);
    save_figure(moment, '1pred')

    plt.figure(figsize=(10, 5))
    for _x, _name in zip((x, restored, y), ('low', 'CARE', 'GT')):
        plt.plot(normalize(_x, 1, 99.7)[180], label=_name, lw=2)
    plt.legend()
    save_figure(moment, 'loss')



if True:
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

if load:
    ### Load images
    #y = imread('data_mito/test/GT/IMG0052.STED.ome.tif')
    #x = imread('data_mito/test/low/IMG0052.STED.ome.tif')
    y = imread('data_care/test/GT/img_0001.tif')
    x = imread('data_care/test/low/img_0002.tif')
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
from __future__ import print_function, unicode_literals, absolute_import, division
from operator import truediv
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
import datetime
import os
import keras

from csbdeep.utils import download_and_extract_zip_file, plot_some, normalize
from csbdeep.data import RawData, create_patches, create_patches_mito, no_background_patches, norm_percentiles, sample_percentiles


from csbdeep.utils import axes_dict, plot_some, plot_history, Path, download_and_extract_zip_file
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
from csbdeep.models import Config, CARE



if True:
    ### Keep time
    date = datetime.datetime.now()
    year = date.month
    day = date.day
    hour = date.hour
    min = date.minute
    moment = str(year) + '-' + str(day) + '_' +  str(hour) + '-' + str(min) + '.jpg'

if True:
    ### Download and extract zip file
    download_and_extract_zip_file (
        url       = 'http://csbdeep.bioimagecomputing.com/example_data/snr_7_binning_2.zip',
        targetdir = 'data',
        verbose   = 1,
    )

    ### Generate training data
    raw_data = RawData.from_folder (
        basepath    = 'data/train',
        source_dirs = ['low'],
        target_dir  = 'GT',
        axes        = 'YX',
    )

    ### Creation of patches
    X, Y, XY_axes = create_patches_mito (
        raw_data            = raw_data,
        patch_size          = (128,128),
        #n_patches_per_image = 2,
        patch_filter        = no_background_patches(0),
        save_file           = 'data/my_training_data_bis.npz',
    )

if True:
    ### Split into training and validation data
    (X,Y), (X_val,Y_val), axes = load_training_data('data/my_training_data_bis.npz', validation_split=0.05, verbose=True)
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

if True:
    ### CARE model
    config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=3, train_batch_size=8, train_steps_per_epoch=40)
    print(config)
    vars(config)
    model = CARE(config, 'my_model', basedir='models')
    model.keras_model.summary()
    history = model.train(X,Y, validation_data=(X_val,Y_val), epochs=10)

    ### Plot history
    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16,5))
    plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);

    if not os.path.isdir('fig/'):
        os.makedirs('fig/')
    file_name_history = 'fig/history_' + moment
    plt.savefig(file_name_history)

if False:
    file = 'models/my_model/weights_last.h5'
    model = keras.models.load_model(file)
    model.compile()
    score = model.evaluate()
    accuracies.append(score[3])
    print("%s: %.2f%%" % (model.metrics_names[0], score[0]))
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]))
    print("%s: %.2f%%" % (model.metrics_names[2], score[2]))
    print("%s: %.2f%%" % (model.metrics_names[3], score[3]))
    print("%s: %.2f%%" % (model.metrics_names[4], score[4]))


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

    if not os.path.isdir('fig/'):
        os.makedirs('fig/')
    file_name_pred5 = 'fig/pred5_' + moment
    plt.savefig(file_name_pred5)
    ### SAVE model
    model.export_TF()
if True:
    ### Load images
    y = imread('data/test/GT/img_0010.tif')
    x = imread('data/test/low/img_0010.tif')
    axes = 'YX'

    ### Load model
    model = CARE(config=None, name='my_model', basedir='models')
    restored = model.predict(x, axes)

    ### Compare original prediction and ground truth
    plt.figure(figsize=(15,10))
    plot_some(np.stack([x,restored,y]),
            title_list=[['low','CARE','GT']], 
            pmin=2,pmax=99.8);
    if not os.path.isdir('fig/'):
        os.makedirs('fig/')
    file_name_pred1 = 'fig/pred1_' + moment
    plt.savefig(file_name_pred1)

    plt.figure(figsize=(10,5))
    for _x,_name in zip((x,restored,y),('low','CARE','GT')):
        plt.plot(normalize(_x,1,99.7)[180], label = _name, lw = 2)

    plt.legend()
    if not os.path.isdir('fig/'):
        os.makedirs('fig/')
    file_name_loss = 'fig/loss_' + moment
    plt.savefig(file_name_loss)
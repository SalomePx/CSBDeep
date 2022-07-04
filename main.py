from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
import argparse
import datetime

import tensorflow as tf
tf.debugging.disable_traceback_filtering()

from csbdeep.utils import normalize, extract_zip_file
from csbdeep.data import RawData, create_patches, create_patches_mito, no_background_patches, norm_percentiles, sample_percentiles, create_test_patches
from csbdeep.data.deteriorate import create_noised_inputs


from csbdeep.utils import axes_dict, plot_some, plot_history, Path, download_and_extract_zip_file, save_figure, normalize_0_255
#from csbdeep.utils.plot_utils import plot_multiple_ssim_maps
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.internals.losses import plot_multiple_ssim_maps
from csbdeep.internals.predict import restore_and_eval_test, restore_and_eval

# -----------------------------
# -------- Time saving ----------
# -----------------------------
def launch(loss):
    from csbdeep.data.transform import flip_vertical, flip_90, flip_180, flip_270, zoom_aug

    date = datetime.datetime.now()
    month = date.month
    day = date.day
    hour = date.hour
    min = date.minute
    sec = date.second
    moment = str(month) + '-' + str(day) + '_' + str(hour) + '-' + str(min) + '_' + str(sec)


    # ----------------------------------------------
    # -------- Global parameters settings ----------
    # ----------------------------------------------
    # Fixed
    initial_care = False
    save_fig = True
    predicting = True
    build_data = True
    base_dir = 'fig/' + moment

    # Non fixed
    create_patches_with_care = False
    data_dir = 'data_mito'
    multiple_maps = False
    fine_tuning = False
    load = False

    # ----------------------------------
    # -------- Plot SSIM maps ----------
    # ----------------------------------

    if multiple_maps:
        name_img = 'IMG0063'
        plot_multiple_ssim_maps(name_img, moment)


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

        # raw_data_test = RawData.from_folder(
        #      basepath=data_dir + '/test',
        #      source_dirs=['low'],
        #      target_dir='GT',
        #      axes='YX',
        # )
        #
        # create_test_patches(raw_data_test, save_dir = data_dir+'/test')

        # Initialization of transforms
        axes = 'YX'
        flip_vertical = flip_vertical(axes)
        flip_90 = flip_90(axes)
        flip_180 = flip_180(axes)
        flip_270 = flip_270(axes)
        zoom_aug = zoom_aug(axes)

        #transforms = [flip_vertical, flip_90, flip_180, flip_270]
        transforms = None

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
                transforms         = transforms,
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



    # ---------------------------------
    # -------- Train the model --------
    # ---------------------------------

    if not load:
        config = Config(axes, n_channel_in, n_channel_out, train_loss=loss, unet_kern_size=3, train_batch_size=8, train_steps_per_epoch=5)
        vars(config)
        model = CARE(config, 'my_model', basedir=base_dir, name_weights='best')
        model.keras_model.summary()
        history = model.train(X, Y, validation_data=(X_val, Y_val), epochs=1)

        # Plot history
        plt.figure(figsize=(16, 5))
        plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mse_focus', 'val_mse_focus', 'mae', 'val_mae'], ['ssim', 'val_ssim','ssim_focus', 'val_ssim_focus'],
                     ['psnr', 'val_psnr', 'psnr_focus', 'val_psnr_focus'])
        save_figure(moment, 'history')


    # ---------------------------------
    # -------- Fine Tuning  -----------
    # ---------------------------------

    if fine_tuning:
        # Load weights automatically inside CARE class
        model_trf = CARE(config=None, name='my_model', basedir='archives/trainings/careWeights', name_weights='weights_best.h5', logdir_save=base_dir)
        layers_care = model_trf.keras_model.layers

        for layer in layers_care[:len(layers_care)-6]:
            layer.trainable = False

        model_trf.keras_model.summary()
        history = model_trf.train(X, Y, validation_data=(X_val, Y_val), epochs=100, steps_per_epoch=50)
        model = model_trf

    # --------------------------------
    # -------- Load the model --------
    # --------------------------------

    #else:
     #   model = CARE(config=None, name='my_model', basedir='archives/trainings/careWeights', name_weights='weights_best.h5')


    # ---------------------------------------------
    # -------- Testings and predictions -----------
    # ---------------------------------------------

    if predicting:
        psnrs, ssims, psnrs_focus, ssims_focus, _, _ = restore_and_eval_test(model, 'YX', data_dir, moment)


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
        plt.figure(figsize=(12, 6))
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
        #idx = [4,5,8,11,15]
        idx = [58,29,60,61,62]
        X_val_set = np.array([X_val[i] for i in idx])
        Y_val_set = np.array([Y_val[i] for i in idx])
        _P = model.keras_model.predict(X_val_set)
        psnrs, ssims = restore_and_eval((Y_val_set, X_val_set, _P))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", action='store', dest='loss', required=True)
    args = parser.parse_args()

    launch(args.loss)




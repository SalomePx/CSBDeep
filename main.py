from __future__ import print_function, unicode_literals, absolute_import, division

import os

import matplotlib.pyplot as plt
from tifffile import imread
import numpy as np
import argparse
import datetime
import cv2

from csbdeep.utils import normalize, extract_zip_file
from csbdeep.data import RawData, create_patches, create_patches_mito, no_background_patches, no_background_patches_zscore
from csbdeep.data.deteriorate import create_noised_inputs

from csbdeep.utils import axes_dict, plot_some, plot_history, Path, download_and_extract_zip_file, save_figure
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.internals.losses import plot_multiple_ssim_maps
from csbdeep.internals.predict import restore_and_eval_test, eval_metrics


def launch(load, loss, transforms, epochs, spe, pred_patch):
    from csbdeep.data.transform import flip_vertical, flip_90, flip_180, flip_270, zoom_aug

    # -----------------------------
    # -------- Time saving ----------
    # -----------------------------
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
    create_patches_with_care = False
    initial_care = False
    save_fig = True
    predicting = True
    build_data = True
    base_dir = 'fig/' + moment
    data_dir = 'data_mito'

    # Goal
    predict_one_img = False
    all_models = False
    multiple_maps = False
    fine_tuning = False
    train = not load

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

        if transforms:
            transforms = [flip_vertical, flip_90, flip_180, flip_270]

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
    elif initial_care:
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
    if train:
        config = Config(axes, n_channel_in, n_channel_out, train_loss=loss, unet_kern_size=3, train_batch_size=8, train_steps_per_epoch=spe)
        vars(config)
        model = CARE(config, 'my_model', basedir=base_dir, name_weights='best')
        model.keras_model.summary()
        history = model.train(X, Y, validation_data=(X_val, Y_val), epochs=epochs)

        # Plot history
        plt.figure(figsize=(16, 5))
        plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mse_focus', 'val_mse_focus', 'mae', 'val_mae'], ['ssim', 'val_ssim','ssim_focus', 'val_ssim_focus'],
                     ['psnr', 'val_psnr', 'psnr_focus', 'val_psnr_focus'])
        save_figure(moment, 'history')


    # --------------------------------
    # -------- Load the model --------
    # --------------------------------
    elif load:
        model = CARE(config=None, name='my_model', basedir='archives/finalTable/'+loss, name_weights='weights_best.h5')


    # ---------------------------------
    # -------- Fine Tuning  -----------
    # ---------------------------------
    elif fine_tuning:
        model_trf = CARE(config=None, name='my_model', basedir='archives/trainings/careWeights', name_weights='weights_best.h5', logdir_save=base_dir)
        layers_care = model_trf.keras_model.layers

        for layer in layers_care[:len(layers_care)-6]:
            layer.trainable = False

        model_trf.keras_model.summary()
        history = model_trf.train(X, Y, validation_data=(X_val, Y_val), epochs=100, steps_per_epoch=50)
        model = model_trf


    # --------------------------------------
    # -------- Predict one image -----------
    # --------------------------------------
    if predict_one_img:
        x = imread('tests/IMG0027.STED.ome.tif')
        if all_models:
            losses = os.listdir('archives/finalTable')
        else:
            losses = [loss]
        for loss in losses:
            model = CARE(config=None, name='my_model', basedir='archives/finalTable/' + loss, name_weights='weights_best.h5')
            restored = model.predict(x, axes= 'YX')

            # Plot GT, prediction and SSIM maps
            '''
            plt.figure(figsize=(15, 10))
            plt.subplot(1, 2, 1)
            plt.title('Original')
            plt.imshow(x)
            plt.subplot(1, 2, 2)
            plt.title('Prediction')
            plt.imshow(restored)
            plt.savefig("tests/IMG0027_" + loss +".png", bbox_inches='tight')'''
            cv2.imwrite("tests/IMG0027_" + loss + ".tif", restored)


    # ---------------------------------------------
    # -------- Testings and predictions -----------
    # ---------------------------------------------
    if predicting:
        if pred_patch == 3 or pred_patch == 1:
            restore_and_eval_test(model, 'YX', data_dir, moment, patch_pred=True)
        if pred_patch == 3 or pred_patch == 2:
            restore_and_eval_test(model, 'YX', data_dir, moment, patch_pred=False)


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
            y = imread(data_dir + '/train/GT/IMG0042.STED.ome.tif')
            x = imread(data_dir + '/train/low/IMG0042.STED.ome.tif')

        # Predict a specific image
        restored = model.predict(x, axes)
        plt.figure(figsize=(12, 6))
        plot_some(np.stack([x, restored, y]), title_list=[['low', 'CARE', 'GT']], pmin=2, pmax=99.8)
        psnrs_f, ssims_f = eval_metrics((y, x), focus=True)
        psnrs_f_low, ssims_f_low = eval_metrics((y, restored), focus=True)
        plt.suptitle(f"Low: PSNR: {round(psnrs_f_low[0], 2)} - SSIM: {round(ssims_f_low[0], 2)}\n"
                     f"Prediction: PSNR: {round(psnrs_f[0], 2)} - SSIM: {round(ssims_f[0], 2)}")
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
        idx = [58, 29, 60, 61, 62]
        X_val_set = np.array([X_val[i] for i in idx])
        Y_val_set = np.array([Y_val[i] for i in idx])
        _P = model.keras_model.predict(X_val_set)
        psnrs_f, ssims_f = eval_metrics((Y_val_set, _P), focus=True)
        plot_some(X_val_set, Y_val_set, _P, pmax=99.5)
        plt.suptitle('5 example validation patches\n'      
                    'top row: input (source),  '          
                    'middle row: target (ground truth),  '
                    'bottom row: predicted from source \n\n'
                     f'1 : PSNR: {round(psnrs_f[0], 2)} - SSIM: {round(ssims_f[0], 2)}             '
                     f'2 : PSNR: {round(psnrs_f[1], 2)} - SSIM: {round(ssims_f[1], 2)}             '
                     f'3 : PSNR: {round(psnrs_f[2], 2)} - SSIM: {round(ssims_f[2], 2)}             '
                     f'4 : PSNR: {round(psnrs_f[3], 2)} - SSIM: {round(ssims_f[3], 2)}             '
                     f'5 : PSNR: {round(psnrs_f[4], 2)} - SSIM: {round(ssims_f[4], 2)}')
        save_figure(moment, '5preds')

        # SAVE model
        model.export_TF()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-load", action='store', dest='load', type=bool, default=False)
    parser.add_argument("-l", action='store', dest='loss', required=True)
    parser.add_argument("-t", action='store', dest='tfm', type=bool, default=False)
    parser.add_argument("-e", action='store', dest='epochs', type=int, default=1)
    parser.add_argument("-spe", action='store', dest='steps_per_epoch', type=int, default=5)
    parser.add_argument("-f", action='store', dest='focus', type=bool, default=False)
    parser.add_argument("-p", action='store', dest='pred_patch', type=int, default=2)  #1: yes 2:no 3:both
    args = parser.parse_args()

    if True:
        launch(args.load, args.loss, args.tfm, args.epochs, args.steps_per_epoch, args.pred_patch)

    if False:
        all_imgs = os.listdir('data_mito/test/low')
        for img in all_imgs:
            x = imread('data_mito/test/low/'+img)
            name = img.split('.')[0]
            no_background_patches_zscore()(x, name)






# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

from itertools import chain
import sys, warnings

from tqdm import tqdm
import tensorflow as tf
import numpy as np
import shutil
import math
import cv2
import os

import torch.nn as nn
import torch

from ..utils import _raise, consume, compose, normalize_mi_ma, axes_dict, axes_check_and_normalize, choice, save_patch, \
    normalize, create_patch_dir, create_dir
from .patch_selection import patch_is_valid_care, patch_is_valid_occupation
from .transform import Transform, permute_axes, broadcast_target

from ..io import save_training_data
from ..utils.six import Path


## Patch filter
def no_background_patches(threshold=0.4, percentile=99.9):
    """Returns a patch filter to be used by :func:`create_patches` to determine for each image pair which patches
    are eligible for sampling. The purpose is to only sample patches from "interesting" regions of the raw image that
    actually contain a substantial amount of non-background signal. To that end, a maximum filter is applied to the target image
    to find the largest values in a region.

    Parameters
    ----------
    threshold : float, optional
        Scalar threshold between 0 and 1 that will be multiplied with the (outlier-robust)
        maximum of the image (see `percentile` below) to denote a lower bound.
        Only patches with a maximum value above this lower bound are eligible to be sampled.
    percentile : float, optional
        Percentile value to denote the (outlier-robust) maximum of an image, i.e. should be close 100.

    Returns
    -------
    function
        Function that takes an image pair `(y,x)` and the patch size as arguments and
        returns a binary mask of the same size as the image (to denote the locations
        eligible for sampling for :func:`create_patches`). At least one pixel of the
        binary mask must be ``True``, otherwise there are no patches to sample.

    Raises
    ------
    ValueError
        Illegal arguments.
    """

    (np.isscalar(percentile) and 0 <= percentile <= 100) or _raise(ValueError())
    (np.isscalar(threshold) and 0 <= threshold <= 1) or _raise(ValueError())

    from scipy.ndimage.filters import maximum_filter
    def _filter(datas, patch_size, dtype=np.float32):
        image = datas[0]
        if dtype is not None:
            image = image.astype(dtype)
        # make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        patch_size = [(p // 2 if p > 1 else p) for p in patch_size]
        filtered = maximum_filter(image, patch_size, mode='constant')
        return filtered > threshold * np.percentile(image, percentile)

    return _filter


def no_background_patches_zscore_old():
    # TODO : the definition : in construction

    """Returns a patch filter to be used by :func:`create_patches` to determine for each image pair which patches
    are eligible for sampling. The purpose is to only sample patches from "interesting" regions of the raw image that
    actually contain a substantial amount of non-background signal. To that end, a maximum filter is applied to the target image
    to find the largest values in a region.

    Returns
    -------
    function
        Function that takes an image pair `(y,x)` and the patch size as arguments and
        returns a binary mask of the same size as the image (to denote the locations
        eligible for sampling for :func:`create_patches`). At least one pixel of the
        binary mask must be ``True``, otherwise there are no patches to sample.

    Raises
    ------
    ValueError
        Illegal arguments.
    """


    def _filter(y, image_name='', dtype=np.float32, save=True):
        if dtype is not None:
            y = y.astype(dtype)

        # Make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        filtered = (y - np.median(y)) / np.std(y)
        filtered = np.where(filtered < 0, 0, filtered)
        mask_filter = filtered > 1

        if save:
            cv2.imwrite("savings/filtered_images/" + image_name + ".tif", filtered)
            zscore_img = np.where(mask_filter != 0, filtered, 0)
            cv2.imwrite("savings/zscore_images/" + image_name + ".tif", zscore_img)

        return mask_filter

    return _filter


def no_background_patches_zscore():
    # TODO : the definition : in construction

    """Returns a patch filter to be used by :func:`create_patches` to determine for each image pair which patches
    are eligible for sampling. The purpose is to only sample patches from "interesting" regions of the raw image that
    actually contain a substantial amount of non-background signal. To that end, a maximum filter is applied to the target image
    to find the largest values in a region.

    Returns
    -------
    function
        Function that takes an image pair `(y,x)` and the patch size as arguments and
        returns a binary mask of the same size as the image (to denote the locations
        eligible for sampling for :func:`create_patches`). At least one pixel of the
        binary mask must be ``True``, otherwise there are no patches to sample.

    Raises
    ------
    ValueError
        Illegal arguments.
    """
    def residual(y):
        down = y[1:, :-1]
        right = y[:-1, 1:]
        res = (2 * y[:-1,:-1] - down - right) / math.sqrt(6)
        return res

    def anscombe_tfm(y):
        y = np.where(y<0, 0, y)
        tfm = 2 * np.sqrt(3/8 + y)
        return tfm

    def mad(y):
        y = np.abs(y)
        m = np.median(y)
        sigma = 1.4826 * m
        return sigma

    def std_approx(y, poisson=True):
        if poisson:
            y = anscombe_tfm(y)
        res = residual(y)
        std = mad(res)
        return std


    def _filter(y, image_name='', dtype=np.float32, save=True):
        if dtype is not None:
            y = y.astype(dtype)

        # Make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        filtered = (y - np.median(y)) / std_approx(y)
        filtered = np.where(filtered < 0, 0, filtered)
        mask_filter = filtered > 1

        if save:
            cv2.imwrite("savings/filtered_images/" + image_name + ".tif", filtered)
            zscore_img = np.where(mask_filter != 0, filtered, 0)
            cv2.imwrite("savings/zscore_images/" + image_name + ".tif", zscore_img)

        return mask_filter

    return _filter


# Sample patches
def sample_patches_from_multiple_stacks(datas, patch_size, n_samples, datas_mask=None, patch_filter=None,
                                        verbose=False):
    """ sample matching patches of size `patch_size` from all arrays in `datas` """

    # TODO: some of these checks are already required in 'create_patches'
    len(patch_size) == datas[0].ndim or _raise(ValueError())

    if not all((a.shape == datas[0].shape for a in datas)):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all((0 < s <= d for s, d in zip(patch_size, datas[0].shape))):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (
            str(patch_size), str(datas[0].shape)))

    if patch_filter is None:
        patch_mask = np.ones(datas[0].shape, dtype=np.bool)
    else:
        patch_mask = patch_filter(datas, patch_size)

    if datas_mask is not None:
        # TODO: Test this
        warnings.warn('Using pixel masks for raw/transformed images not tested.')
        datas_mask.shape == datas[0].shape or _raise(ValueError())
        datas_mask.dtype == np.bool or _raise(ValueError())
        from scipy.ndimage.filters import minimum_filter
        patch_mask &= minimum_filter(datas_mask, patch_size, mode='constant', cval=False)

    # get the valid indices
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, datas[0].shape)])
    valid_inds = np.where(patch_mask[border_slices])
    n_valid = len(valid_inds[0])

    if n_valid == 0:
        raise ValueError("'patch_filter' didn't return any region to sample from")

    sample_inds = choice(range(n_valid), n_samples, replace=(n_valid < n_samples))
    rand_inds = [v[sample_inds] + s.start for s, v in zip(border_slices, valid_inds)]

    res = [np.stack([data[tuple(slice(_r - (_p // 2), _r + _p - (_p // 2)) for _r, _p in zip(r, patch_size))] for r in
                     zip(*rand_inds)]) for data in datas]
    return res


# Create training data
def _valid_low_high_percentiles(ps):
    return isinstance(ps, (list, tuple, np.ndarray)) and len(ps) == 2 and all(map(np.isscalar, ps)) and (
            0 <= ps[0] < ps[1] <= 100)


def _memory_check(n_required_memory_bytes, thresh_free_frac=0.5, thresh_abs_bytes=1024 * 1024 ** 2):
    try:
        # raise ImportError
        import psutil
        mem = psutil.virtual_memory()
        mem_frac = n_required_memory_bytes / mem.available
        if mem_frac > 1:
            raise MemoryError('Not enough available memory.')
        elif mem_frac > thresh_free_frac:
            print('Warning: will use at least %.0f MB (%.1f%%) of available memory.\n' % (
                n_required_memory_bytes / 1024 ** 2, 100 * mem_frac), file=sys.stderr)
            sys.stderr.flush()
    except ImportError:
        if n_required_memory_bytes > thresh_abs_bytes:
            print('Warning: will use at least %.0f MB of memory.\n' % (n_required_memory_bytes / 1024 ** 2),
                  file=sys.stderr)
            sys.stderr.flush()


def sample_percentiles(pmin=(1, 3), pmax=(99.5, 99.9)):
    """Sample percentile values from a uniform distribution.

    Parameters
    ----------
    pmin : tuple
        Tuple of two values that denotes the interval for sampling low percentiles.
    pmax : tuple
        Tuple of two values that denotes the interval for sampling high percentiles.

    Returns
    -------
    function
        Function without arguments that returns `(pl,ph)`, where `pl` (`ph`) is a sampled low (high) percentile.

    Raises
    ------
    ValueError
        Illegal arguments.
    """
    _valid_low_high_percentiles(pmin) or _raise(ValueError(pmin))
    _valid_low_high_percentiles(pmax) or _raise(ValueError(pmax))
    pmin[1] < pmax[0] or _raise(ValueError())
    return lambda: (np.random.uniform(*pmin), np.random.uniform(*pmax))


def norm_percentiles(percentiles=sample_percentiles(), relu_last=False):
    """Normalize extracted patches based on percentiles from corresponding raw image.

    Parameters
    ----------
    percentiles : tuple, optional
        A tuple (`pmin`, `pmax`) or a function that returns such a tuple, where the extracted patches
        are (affinely) normalized in such that a value of 0 (1) corresponds to the `pmin`-th (`pmax`-th) percentile
        of the raw image (default: :func:`sample_percentiles`).
    relu_last : bool, optional
        Flag to indicate whether the last activation of the CARE network is/will be using
        a ReLU activation function (default: ``False``)

    Return
    ------
    function
        Function that does percentile-based normalization to be used in :func:`create_patches`.

    Raises
    ------
    ValueError
        Illegal arguments.

    Todo
    ----
    ``relu_last`` flag problematic/inelegant.

    """
    if callable(percentiles):
        _tmp = percentiles()
        _valid_low_high_percentiles(_tmp) or _raise(ValueError(_tmp))
        get_percentiles = percentiles
    else:
        _valid_low_high_percentiles(percentiles) or _raise(ValueError(percentiles))
        get_percentiles = lambda: percentiles

    def _normalize(patches_x, patches_y, x, y, mask, channel):
        pmins, pmaxs = zip(*(get_percentiles() for _ in patches_x))
        percentile_axes = None if channel is None else tuple((d for d in range(x.ndim) if d != channel))
        _perc = lambda a, p: np.percentile(a, p, axis=percentile_axes, keepdims=True)
        patches_x_norm = normalize_mi_ma(patches_x, _perc(x, pmins), _perc(x, pmaxs))
        if relu_last:
            pmins = np.zeros_like(pmins)
        patches_y_norm = normalize_mi_ma(patches_y, _perc(y, pmins), _perc(y, pmaxs))
        return patches_x_norm, patches_y_norm

    return _normalize


def create_patches(
        raw_data,
        patch_size,
        n_patches_per_image,
        patch_axes=None,
        save_file=None,
        transforms=None,
        patch_filter=no_background_patches(),
        normalization=norm_percentiles(),
        shuffle=True,
        verbose=True,
):
    """Create normalized training data to be used for neural network training.

    Parameters
    ----------
    raw_data : :class:`RawData`
        Object that yields matching pairs of raw images.
    patch_size : tuple
        Shape of the patches to be extraced from raw images.
        Must be compatible with the number of dimensions and axes of the raw images.
        As a general rule, use a power of two along all XYZT axes, or at least divisible by 8.
    n_patches_per_image : int
        Number of patches to be sampled/extracted from each raw image pair (after transformations, see below).
    patch_axes : str or None
        Axes of the extracted patches. If ``None``, will assume to be equal to that of transformed raw data.
    save_file : str or None
        File name to save training data to disk in ``.npz`` format (see :func:`csbdeep.io.save_training_data`).
        If ``None``, data will not be saved.
    transforms : list or tuple, optional
        List of :class:`Transform` objects that apply additional transformations to the raw images.
        This can be used to augment the set of raw images (e.g., by including rotations).
        Set to ``None`` to disable. Default: ``None``.
    patch_filter : function, optional
        Function to determine for each image pair which patches are eligible to be extracted
        (default: :func:`no_background_patches`). Set to ``None`` to disable.
    normalization : function, optional
        Function that takes arguments `(patches_x, patches_y, x, y, mask, channel)`, whose purpose is to
        normalize the patches (`patches_x`, `patches_y`) extracted from the associated raw images
        (`x`, `y`, with `mask`; see :class:`RawData`). Default: :func:`norm_percentiles`.
    shuffle : bool, optional
        Randomly shuffle all extracted patches.
    verbose : bool, optional
        Display overview of images, transforms, etc.

    Returns
    -------
    tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, str)
        Returns a tuple (`X`, `Y`, `axes`) with the normalized extracted patches from all (transformed) raw images
        and their axes.
        `X` is the array of patches extracted from source images with `Y` being the array of corresponding target patches.
        The shape of `X` and `Y` is as follows: `(n_total_patches, n_channels, ...)`.
        For single-channel images, `n_channels` will be 1.

    Raises
    ------
    ValueError
        Various reasons.

    Example
    -------
    >>> raw_data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='ZYX')
    >>> X, Y, XY_axes = create_patches(raw_data, patch_size=(32,128,128), n_patches_per_image=16)

    Todo
    ----
    - Save created patches directly to disk using :class:`numpy.memmap` or similar?
      Would allow to work with large data that doesn't fit in memory.

    """
    # Images and transforms
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    if patch_axes is not None:
        transforms.append(permute_axes(patch_axes))
    if len(transforms) == 0:
        transforms.append(Transform.identity())

    if normalization is None:
        normalization = lambda patches_x, patches_y, x, y, mask, channel: (patches_x, patches_y)

    image_pairs, n_raw_images = raw_data.generator(), raw_data.size
    tf = Transform(*zip(*transforms))
    image_pairs = compose(*tf.generator)(image_pairs)
    n_transforms = np.prod(tf.size)
    n_images = n_raw_images * n_transforms
    n_patches = n_images * n_patches_per_image
    n_required_memory_bytes = 2 * n_patches * np.prod(patch_size) * 4

    # Memory check
    _memory_check(n_required_memory_bytes)

    # Summary
    if verbose:
        print('=' * 66)
        print('%5d raw images x %4d transformations   = %5d images' % (n_raw_images, n_transforms, n_images))
        print('%5d images     x %4d patches per image = %5d patches in total' % (
            n_images, n_patches_per_image, n_patches))
        print('=' * 66)
        print('Input data:')
        print(raw_data.description)
        print('=' * 66)
        print('Transformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('=' * 66)
        print('Patch size:')
        print(" x ".join(str(p) for p in patch_size))
        print('=' * 66)

    sys.stdout.flush()

    # Sample patches from each pair of transformed raw images
    X = np.empty((n_patches,) + tuple(patch_size), dtype=np.float32)
    Y = np.empty_like(X)

    for i, (x, y, _axes, mask) in tqdm(enumerate(image_pairs), total=n_images, disable=(not verbose)):
        if i >= n_images:
            warnings.warn('more raw images (or transformations thereof) than expected, skipping excess images.')
            break
        if i == 0:
            axes = axes_check_and_normalize(_axes, len(patch_size))
            channel = axes_dict(axes)['C']
        # checks
        # len(axes) >= x.ndim or _raise(ValueError())
        axes == axes_check_and_normalize(_axes) or _raise(ValueError('not all images have the same axes.'))
        x.shape == y.shape or _raise(ValueError())
        mask is None or mask.shape == x.shape or _raise(ValueError())
        (channel is None or (isinstance(channel, int) and 0 <= channel < x.ndim)) or _raise(ValueError())
        channel is None or patch_size[channel] == x.shape[channel] or _raise(
            ValueError('extracted patches must contain all channels.'))

        _Y, _X = sample_patches_from_multiple_stacks((y, x), patch_size, n_patches_per_image, mask, patch_filter)

        s = slice(i * n_patches_per_image, (i + 1) * n_patches_per_image)
        X[s], Y[s] = normalization(_X, _Y, x, y, mask, channel)

    if shuffle:
        shuffle_inplace(X, Y)

    axes = 'SC' + axes.replace('C', '')
    if channel is None:
        X = np.expand_dims(X, 1)
        Y = np.expand_dims(Y, 1)
    else:
        X = np.moveaxis(X, 1 + channel, 1)
        Y = np.moveaxis(Y, 1 + channel, 1)

    if save_file is not None:
        print('Saving data to %s.' % str(Path(save_file)))
        save_training_data(save_file, X, Y, axes)

    return X, Y, axes


def create_patches_reduced_target(
        raw_data,
        patch_size,
        n_patches_per_image,
        reduction_axes,
        target_axes=None,  # TODO: this should rather be part of RawData and also exposed to transforms
        **kwargs
):
    """Create normalized training data to be used for neural network training.

    In contrast to :func:`create_patches`, it is assumed that the target image has reduced
    dimensionality (i.e. size 1) along one or several axes (`reduction_axes`).

    Parameters
    ----------
    raw_data : :class:`RawData`
        See :func:`create_patches`.
    patch_size : tuple
        See :func:`create_patches`.
    n_patches_per_image : int
        See :func:`create_patches`.
    reduction_axes : str
        Axes where the target images have a reduced dimension (i.e. size 1) compared to the source images.
    target_axes : str
        Axes of the raw target images. If ``None``, will be assumed to be equal to that of the raw source images.
    kwargs : dict
        Additional parameters as in :func:`create_patches`.

    Returns
    -------
    tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, str)
        See :func:`create_patches`. Note that the shape of the target data will be 1 along all reduction axes.

    """
    reduction_axes = axes_check_and_normalize(reduction_axes, disallowed='S')

    transforms = kwargs.get('transforms')
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    transforms.insert(0, broadcast_target(target_axes))
    kwargs['transforms'] = transforms

    save_file = kwargs.pop('save_file', None)

    if any(s is None for s in patch_size):
        patch_axes = kwargs.get('patch_axes')
        if patch_axes is not None:
            _transforms = list(transforms)
            _transforms.append(permute_axes(patch_axes))
        else:
            _transforms = transforms
        tf = Transform(*zip(*_transforms))
        image_pairs = compose(*tf.generator)(raw_data.generator())
        x, y, axes, mask = next(image_pairs)  # get the first entry from the generator
        patch_size = list(patch_size)
        for i, (a, s) in enumerate(zip(axes, patch_size)):
            if s is not None: continue
            a in reduction_axes or _raise(ValueError("entry of patch_size is None for non reduction axis %s." % a))
            patch_size[i] = x.shape[i]
        patch_size = tuple(patch_size)
        del x, y, axes, mask

    X, Y, axes = create_patches(
        raw_data=raw_data,
        patch_size=patch_size,
        n_patches_per_image=n_patches_per_image,
        **kwargs
    )

    ax = axes_dict(axes)
    for a in reduction_axes:
        a in axes or _raise(ValueError("reduction axis %d not present in extracted patches" % a))
        n_dims = Y.shape[ax[a]]
        if n_dims == 1:
            warnings.warn("extracted target patches already have dimensionality 1 along reduction axis %s." % a)
        else:
            t = np.take(Y, (1,), axis=ax[a])
            Y = np.take(Y, (0,), axis=ax[a])
            i = np.random.choice(Y.size, size=100)
            if not np.all(t.flat[i] == Y.flat[i]):
                warnings.warn("extracted target patches vary along reduction axis %s." % a)

    if save_file is not None:
        print('Saving data to %s.' % str(Path(save_file)))
        save_training_data(save_file, X, Y, axes)

    return X, Y, axes


# Misc

def shuffle_inplace(*arrs, **kwargs):
    seed = kwargs.pop('seed', None)
    if seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(seed=seed)
    state = rng.get_state()
    for a in arrs:
        rng.set_state(state)
        rng.shuffle(a)


# ---------------------------------------------------------------------------------------------------------
# ----------------------------------------- MITOCHONDRIA PART ---------------------------------------------
# ---------------------------------------------------------------------------------------------------------

def cut_patches_in_image(datas, patch_size, image_name='', delete_black_patches=True, occup_min=0.02, overlap=0):
    """Cut images in patches with the method of a grid.

    Use the previous calculation of how many patches can fulfill the height and width (round to ceil), leading to
    overlap between the two last patches. If patch size is (128,128) and image is (300, 300), it will contain
    (h, w) = (300 // 128 + 1, 300 // 128) patches for height and width respectively and a total of h*w patches.

    Parameters
    ----------
    datas : tuple of numpy array (y, x)
        Where y is GT and x is input
    patch_size : tuple of int (a, b)
        Where a is the size of the x-axis and b is the size of the y-axis
    image_name : str
        Name of the image in which we cut patches (to save patches)
    delete_black_patches : bool
        True if we delete background patches with the z-score threshold
    occup_min : float between 0 and 1
        Minimum occupation of relevant information. To adjust according to the dataset.
        For the selection of patches. Used with delete_black_patches=True only.
    Returns
    -------
    tuple(y:class:`numpy.ndarray`, x:class:`numpy.ndarray`)
        Each element of the tuple contains the cutted patches of GT and low resolution input respectively
    """

    y, x = datas
    patches_x = []
    patches_y = []
    nb_saved_patch = 1
    nb_patch = 1

    n_height = math.ceil(x.shape[0]/ patch_size[0])
    new_shape_x = x.shape[0] + (n_height - 1) * overlap
    n_height = math.ceil(new_shape_x/ patch_size[0])

    n_width = math.ceil(x.shape[1] / patch_size[1])
    new_shape_y = x.shape[1] + (n_width - 1) * overlap
    n_width = math.ceil(new_shape_y / patch_size[0])

    # Checks
    assert (each_patch_size > 0 and (type(each_patch_size) is int) for each_patch_size in patch_size)
    assert x.shape == y.shape and x.shape[0] >= patch_size[0] and x.shape[1] >= patch_size[1]
    assert type(image_name) is str
    assert n_height > 0 and n_width > 0
    assert 0 <= occup_min <= 1


    # Normalization of data
    x_norm = x * 255.0 / x.max()
    y_norm = y * 255.0 / y.max()

    # Apply filter (Z-score) on images for detection of black filters if wanted
    if delete_black_patches:
        filtered = (y_norm - np.median(y_norm)) / np.std(y_norm)
        filtered = np.where(filtered < 0, 0, filtered)
        y_filter = np.where(filtered > 1)
        cv2.imwrite("savings/filtered_images/" + image_name + ".tif", filtered)

    # Create patches
    for i in range(n_height):
        for j in range(n_width):
            end_height = i * patch_size[0] + patch_size[0] - overlap * i
            end_width = j * patch_size[1] + patch_size[1] - overlap * j

            # We are in the bottom right corner of the image
            if end_height > x.shape[0] and end_width > x.shape[1]:
                start_height = x.shape[0] - patch_size[0]
                start_width = x.shape[1] - patch_size[1]
                end_height = x.shape[0]
                end_width = x.shape[1]

            # We are on the bottom of the image
            elif end_height > x.shape[0]:
                start_width = j * (patch_size[1] - overlap)
                start_height = x.shape[0] - patch_size[0]
                end_width = start_width + patch_size[1]
                end_height = x.shape[0]

            # We are on the right of the image
            elif end_width > x.shape[1]:
                start_height = i * (patch_size[0] - overlap)
                start_width = x.shape[1] - patch_size[1]
                end_height = start_height + patch_size[0]
                end_width = x.shape[1]

            # We do not touch an end corner of the image
            else:
                start_height = i * (patch_size[0] - overlap)
                end_height = start_height + patch_size[0]
                start_width = j * (patch_size[1] - overlap)
                end_width = start_width + patch_size[1]

            # Finally
            patch_x, patch_y = x[start_height:end_height, start_width:end_width], y[start_height:end_height,
                                                                                  start_width:end_width]
            patch_x_norm, patch_y_norm = x_norm[start_height:end_height, start_width:end_width], y_norm[start_height:end_height,
                                                                                  start_width:end_width]
            if delete_black_patches:
                patch_y_bool = y_filter[start_height:end_height, start_width:end_width]
                patch_y_filter = filtered[start_height:end_height, start_width:end_width]

            # Keep patch or not
            if delete_black_patches:
                ys = [patch_y_norm, patch_y_bool, patch_y_filter]
                admission = patch_is_valid_occupation(ys, image_name, nb_patch, occup_min)
            else:
                admission = True

            # Add the patch in the general list if admissible
            if admission:
                patches_x.append(patch_x.tolist())
                patches_y.append(patch_y.tolist())

                # Saving patches in folder patches
                patch_file_name = image_name + '_' + str(nb_saved_patch) + '.tif'
                save_patch(patch_file_name, (patch_x, patch_y))
                nb_saved_patch += 1

            nb_patch += 1

    return np.array(patches_y), np.array(patches_x)


def split_tensor(image, tile_size=128, overlap=20):
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, image.ndim)
    tensor = torch.from_numpy(image)
    tensor = tensor.permute(3, 0, 1, 2)
    mask = torch.ones_like(tensor)
    stride = tile_size - overlap
    unfold = nn.Unfold(kernel_size=(tile_size, tile_size), stride=stride)
    mask_p = unfold(mask)
    patches = unfold(tensor)

    patches = patches.reshape(1, tile_size, tile_size, -1).permute(3, 0, 1, 2)
    if tensor.is_cuda:
        patches_base = torch.zeros(patches.size(), device=tensor.get_device())
    else:
        patches_base = torch.zeros(patches.size())

    tiles = []
    for t in range(patches.size(0)):
        tiles.append(patches[[t], :, :, :])
    return tiles, mask_p, patches_base, (tensor.size(2), tensor.size(3))


def rebuild_tensor(tensor_list, mask_t, base_tensor, t_size, tile_size=128, overlap=20):
    stride = tile_size - overlap
    for t, tile in enumerate(tensor_list):
        base_tensor[[t], :, :] = tile

    base_tensor = base_tensor.permute(1, 2, 3, 0).reshape(1 * tile_size * tile_size, base_tensor.size(0))
    fold = nn.Fold(output_size=(t_size[0], t_size[1]), kernel_size=(tile_size, tile_size), stride=stride)
    output_tensor = fold(base_tensor) / fold(mask_t)

    output_tensor = output_tensor.permute(1, 2, 3, 0)
    output_array = output_tensor.numpy()
    output_tensor = tf.convert_to_tensor(output_array)
    output = output_tensor.numpy().squeeze().squeeze()
    return output



def create_patches_mito(
        raw_data,
        patch_size,
        data_path,
        patch_axes=None,
        transforms=None,
        cut_or_sample_patch='cut',
        delete_black_patches=True,
        save_file=None,
        normalization=norm_percentiles(),
        shuffle=False,
        verbose=True):
    """Create normalized training data to be used for neural network training.

    Parameters
    ----------
    raw_data : :class:`RawData`
        Object that yields matching pairs of raw images.
    patch_axes : str or None
        Axes of the extracted patches. If ``None``, will assume to be equal to that of transformed raw data.
    data_path : str
        Path of the training collected data
    patch_size : tuple
        Shape of the patches to be extracted from raw images.
        Must be compatible with the number of dimensions and axes of the raw images.
        As a general rule, use a power of two along all XYZT axes, or at least divisible by 8.
    save_file : str or None
        File name to save training data to disk in ``.npz`` format (see :func:`csbdeep.io.save_training_data`).
        If ``None``, data will not be saved.
    cut_or_sample_patch : 'cut' or 'sample'
        Define whether we chose to select random patches or to cut them as a grid within images.
    transforms : list or tuple, optional
        List of :class:`Transform` objects that apply additional transformations to the raw images.
        This can be used to augment the set of raw images (e.g., by including rotations).
        Set to ``None`` to disable. Default: ``None``.
    delete_black_patches : function, optional
        Function to determine for each image pair which patches are eligible to be extracted
        (default: :func:`no_background_patches`). Set to ``None`` to disable.
    normalization : function, optional
        Function that takes arguments `(patches_x, patches_y, x, y, mask, channel)`, whose purpose is to
        normalize the patches (`patches_x`, `patches_y`) extracted from the associated raw images
        (`x`, `y`, with `mask`; see :class:`RawData`). Default: :func:`norm_percentiles`.
    shuffle : bool, optional
        Randomly shuffle all extracted patches.
    verbose : bool, optional
        Display overview of images, transforms, etc.

    Returns
    -------
    tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        Returns a tuple (`X`, `Y`, `axes`) with the normalized extracted patches from all (transformed) raw images
        and their axes.
        `X` is the array of patches extracted from source images with `Y` being the array of corresponding target patches.
        The shape of `X` and `Y` is as follows: `(n_total_patches, n_channels, ...)`.
        For single-channel images, `n_channels` will be 1.

    Raises
    ------
    ValueError
        Various reasons.
    """

    # Images and transforms
    if not transforms:
        transforms = []
    transforms = list(transforms)
    if patch_axes is not None:
        transforms.append(permute_axes(patch_axes))
    transforms.append(Transform.identity())

    if normalization is None:
        normalization = lambda patches_x, patches_y, x, y, mask, channel: (patches_x, patches_y)

    nb_transforms = len(transforms)
    all_files_names = sorted(os.listdir(data_path + '/train/GT'))
    initial_nb_images = raw_data.size

    # Create one generator of generators for all transforms
    all_image_pairs = [raw_data.generator() for _ in range(nb_transforms)]
    tf = Transform(*zip(*transforms))
    generators = [gen(image_pair) for gen, image_pair in zip(tf.generator, all_image_pairs)]
    image_pairs = chain(*generators)

    # Summary
    if verbose:
        print('=' * 66)
        print('Input data:')
        print(raw_data.description)
        print('=' * 66)
        print('Transformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('=' * 66)
        print('Patch size:')
        print(" x ".join(str(p) for p in patch_size))
        print('=' * 66)
    sys.stdout.flush()

    # Calculate number of final images
    n_patches = 0
    list_image_pair = list(enumerate(image_pairs))
    image_pair_iter = list_image_pair[:]
    for _, (x, y, _axes, mask) in image_pair_iter:

        if x.shape[0] >= patch_size[0] and x.shape[1] >= patch_size[1]:
            # Calculate the number of patches per image and total final images
            n_patches_height = math.ceil(x.shape[0] / patch_size[0])
            n_patches_width = math.ceil(x.shape[1] / patch_size[1])
            n_patch_per_image = n_patches_height * n_patches_width
            n_patches += n_patch_per_image

    # Memory check
    n_required_memory_bytes = n_patches * np.prod(patch_size)
    _memory_check(n_required_memory_bytes)

    # Initialize sampling patches from each pair of transformed raw images
    X = np.empty((n_patches,) + tuple(patch_size), dtype=np.float32)
    Y = np.empty_like(X)

    # Create directory where to save files
    create_dir('savings')
    create_patch_dir('savings/patches')
    create_dir('savings/deleted_patches')
    create_dir('savings/filtered_images')
    create_dir('savings/zscore_images')

    # Create patches for each image
    print("Building data patches:")
    occupied = 0
    for i, (x, y, _axes, mask) in tqdm(list_image_pair, total=len(list_image_pair), disable=(not verbose)):

        idx = i % initial_nb_images
        tfm = i // initial_nb_images
        image_name = all_files_names[idx].split('.')[0] + '_' + str(tfm)

        if i == 0:
            axes = axes_check_and_normalize(_axes, len(patch_size))
            channel = axes_dict(axes)['C']

        # Checks
        axes == axes_check_and_normalize(_axes) or _raise(ValueError('not all images have the same axes.'))
        x.shape == y.shape or _raise(ValueError())
        mask is None or mask.shape == x.shape or _raise(ValueError())
        (channel is None or (isinstance(channel, int) and 0 <= channel < x.ndim)) or _raise(ValueError())
        channel is None or patch_size[channel] == x.shape[channel] or _raise(
            ValueError('extracted patches must contain all channels.'))

        # If image is big enough
        if x.shape[0] >= patch_size[0] and x.shape[1] >= patch_size[1]:
            # Calculate number of patcher per image
            n_patches_height = math.ceil(x.shape[0] / patch_size[0])
            n_patches_width = math.ceil(x.shape[1] / patch_size[1])

            # Create patches
            # Create with grid method
            if cut_or_sample_patch != 'cut':
                n_samples = (n_patches_height * n_patches_width)
                _Y, _X = sample_patches_in_image((y, x), patch_size, n_samples, image_name)
            # Create with random selection among interest areas
            else:
                _Y, _X = cut_patches_in_image((y, x), patch_size, image_name, delete_black_patches)
            n_patches_per_image = len(_X)

            # Add the selected patches if at least on patch is kept within the image
            if n_patches_per_image != 0:
                s = slice(occupied, occupied + n_patches_per_image)
                X[s], Y[s] = normalization(_X, _Y, x, y, mask, channel)

            occupied += n_patches_per_image

    X = X[:occupied]
    Y = Y[:occupied]

    if shuffle:
        shuffle_inplace(X, Y)

    axes = 'SC' + axes.replace('C', '')
    if channel is None:
        X = np.expand_dims(X, 1)
        Y = np.expand_dims(Y, 1)
    else:
        X = np.moveaxis(X, 1 + channel, 1)
        Y = np.moveaxis(Y, 1 + channel, 1)

    if save_file is not None:
        print('Saving data to %s.' % str(Path(save_file)))
        save_training_data(save_file, X, Y, axes)

    if verbose:
        print(f"There are {len(X)} final images for training and validation.")
        print('=' * 66)

    return X, Y, axes


def sample_patches_in_image(datas, patch_size, n_samples, image_name, patch_filter=no_background_patches_zscore()):
    """Sample patches in image with the method of areas of interest focusing, containing mitochondria.

    Use the previous calculation of how many patches can fulfill the height and width, to approximate the number of
    patches that can be reasonably generated, to avoid patches repetition.

    Parameters
    ----------
    datas : tuple of numpy array (y, x)
        Where y is GT and x is input
    patch_size : tuple of int (a, b)
        Where a is the size of the x-axis and b is the size of the y-axis
    n_samples : int
        Number of patches created for this image, according to its size (previously calculated).
    image_name : str
        Name of the image in which we cut patches (to save patches)
    patch_filter : function
        What function is used to select good patches and avoid background. Here used Z-score.

    Returns
    -------
    tuple(y:class:`numpy.ndarray`, x:class:`numpy.ndarray`)
        Each element of the tuple contains the sampled patches of GT and low resolution input respectively
    """

    y, x = datas
    if patch_filter is None:
        # Choose to select patches among the whole image
        area_of_info = np.ones(y.shape, dtype=np.bool)
    else:
        # Choose to select patches among specific areas of interest
        area_of_info = patch_filter(y, image_name)

    # Checks
    assert (each_patch_size > 0 and (type(each_patch_size) is int) for each_patch_size in patch_size)
    assert x.shape == y.shape and x.shape[0] >= patch_size[0] and x.shape[1] >= patch_size[1]
    assert type(image_name) is str
    assert n_samples > 0

    # Delineate the zone where the center of future selected patches is possible (with a margin of 1)
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in  zip(patch_size, y.shape)])  # This zone is the center of the image
    # Keep areas of interest only in the center of the image for the selection of patches
    valid_center_idx = np.where(area_of_info[border_slices])

    # There is no relevant information in the image
    n_valid = len(valid_center_idx[0])
    if n_valid == 0:
        raise ValueError("'patch_filter' didn't return any region to sample from")

    # Chose randomly n_samples center points in areas of interest
    sample_idx = choice(range(n_valid), n_samples, replace=(n_valid < n_samples))
    # Obtain the indices of the center points of the selected patch in the global image
    global_center_idx = [v[sample_idx] + s.start for s, v in zip(border_slices, valid_center_idx)]
    # Construct the associated patches for (y,x)
    patches = [np.stack([data[tuple(slice(_r - (_p // 2), _r + _p - (_p // 2)) for _r, _p in zip(r, patch_size))] for r in
                  zip(*global_center_idx)]) for data in datas]
    # _r in the center of the patch
    # _p is the size of the patch
    # _r - (_p // 2) is the index of the beginning of the patch
    # _r + _p - (_p // 2) is the index of the end of the patch

    # Saving patches
    y_res, x_res = patches
    for i in range(len(y_res)):
        save_name = image_name + '_' + str(i) + '.tif'
        save_patch(save_name, (x_res[i], y_res[i]))

    return patches


def create_test_patches(raw_data, save_dir, channel=2):

    all_files_names = sorted(os.listdir(save_dir + '/GT'))

    os.mkdir(save_dir + '/GT2')
    os.mkdir(save_dir + '/low2')

    image_pairs = raw_data.generator()

    for i, (x, y, _, _) in enumerate(image_pairs):
        patches_test_y, patches_test_x = cut_patches_in_image((y, x), patch_size=(128, 128), delete_black_patches=False)
        patches_test_x, patches_test_y = norm_percentiles()(patches_test_x, patches_test_y, x, y, None, channel)
        for j, (patch_y, patch_x) in enumerate(zip(patches_test_y, patches_test_x)):
            image_name = str(all_files_names[i].split('.')[0]) + '_' + str(j) + '.tif'
            cv2.imwrite(save_dir + '/GT2/' + image_name, patch_y)
            cv2.imwrite(save_dir + '/low2/' + image_name, patch_x)

    shutil.rmtree(save_dir + '/GT')
    shutil.rmtree(save_dir + '/low')
    os.rename(save_dir + '/GT2', save_dir + '/GT')
    os.rename(save_dir + '/low2', save_dir + '/low')
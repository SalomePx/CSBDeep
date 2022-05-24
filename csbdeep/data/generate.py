# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

from itertools import chain
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import sys, os, warnings

from tqdm import tqdm
import numpy as np
import shutil
from PIL import Image
import math
import cv2

from ..utils import _raise, consume, compose, normalize_mi_ma, axes_dict, axes_check_and_normalize, choice, save_patch, normalize, create_patch_dir, create_histo_dir
from .transform import Transform, permute_axes, broadcast_target
from ..utils.six import Path
from ..io import save_training_data


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
    (np.isscalar(threshold)  and 0 <= threshold  <=   1) or _raise(ValueError())

    from scipy.ndimage.filters import maximum_filter
    def _filter(datas, patch_size, dtype=np.float32):
        image = datas[0]
        if dtype is not None:
            image = image.astype(dtype)
        # make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        patch_size = [(p//2 if p>1 else p) for p in patch_size]
        filtered = maximum_filter(image, patch_size, mode='constant')
        return filtered > threshold * np.percentile(image,percentile)
    return _filter



## Sample patches

def sample_patches_from_multiple_stacks(datas, patch_size, n_samples, datas_mask=None, patch_filter=None, verbose=False):
    """ sample matching patches of size `patch_size` from all arrays in `datas` """

    # TODO: some of these checks are already required in 'create_patches'
    len(patch_size)==datas[0].ndim or _raise(ValueError())

    if not all(( a.shape == datas[0].shape for a in datas )):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all(( 0 < s <= d for s,d in zip(patch_size,datas[0].shape) )):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (str(patch_size), str(datas[0].shape)))

    if patch_filter is None:
        patch_mask = np.ones(datas[0].shape,dtype=np.bool)
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

    res = [np.stack([data[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,patch_size))] for r in zip(*rand_inds)]) for data in datas]
    return res



## Create training data

def _valid_low_high_percentiles(ps):
    return isinstance(ps,(list,tuple,np.ndarray)) and len(ps)==2 and all(map(np.isscalar,ps)) and (0<=ps[0]<ps[1]<=100)

def _memory_check(n_required_memory_bytes, thresh_free_frac=0.5, thresh_abs_bytes=1024*1024**2):
    try:
        # raise ImportError
        import psutil
        mem = psutil.virtual_memory()
        mem_frac = n_required_memory_bytes / mem.available
        if mem_frac > 1:
            raise MemoryError('Not enough available memory.')
        elif mem_frac > thresh_free_frac:
            print('Warning: will use at least %.0f MB (%.1f%%) of available memory.\n' % (n_required_memory_bytes/1024**2,100*mem_frac), file=sys.stderr)
            sys.stderr.flush()
    except ImportError:
        if n_required_memory_bytes > thresh_abs_bytes:
            print('Warning: will use at least %.0f MB of memory.\n' % (n_required_memory_bytes/1024**2), file=sys.stderr)
            sys.stderr.flush()

def sample_percentiles(pmin=(1,3), pmax=(99.5,99.9)):
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

    def _normalize(patches_x,patches_y, x,y,mask,channel):
        pmins, pmaxs = zip(*(get_percentiles() for _ in patches_x))
        percentile_axes = None if channel is None else tuple((d for d in range(x.ndim) if d != channel))
        _perc = lambda a,p: np.percentile(a,p,axis=percentile_axes,keepdims=True)
        patches_x_norm = normalize_mi_ma(patches_x, _perc(x,pmins), _perc(x,pmaxs))
        if relu_last:
            pmins = np.zeros_like(pmins)
        patches_y_norm = normalize_mi_ma(patches_y, _perc(y,pmins), _perc(y,pmaxs))
        return patches_x_norm, patches_y_norm

    return _normalize


def create_patches(
        raw_data,
        patch_size,
        n_patches_per_image,
        patch_axes    = None,
        save_file     = None,
        transforms    = None,
        patch_filter  = no_background_patches(),
        normalization = norm_percentiles(),
        shuffle       = True,
        verbose       = True,
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
    ## images and transforms
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
    tf = Transform(*zip(*transforms)) # convert list of Transforms into Transform of lists
    image_pairs = compose(*tf.generator)(image_pairs) # combine all transformations with raw images as input
    n_transforms = np.prod(tf.size)
    n_images = n_raw_images * n_transforms
    n_patches = n_images * n_patches_per_image
    n_required_memory_bytes = 2 * n_patches*np.prod(patch_size) * 4

    ## memory check
    _memory_check(n_required_memory_bytes)

    ## summary
    if verbose:
        print('='*66)
        print('%5d raw images x %4d transformations   = %5d images' % (n_raw_images,n_transforms,n_images))
        print('%5d images     x %4d patches per image = %5d patches in total' % (n_images,n_patches_per_image,n_patches))
        print('='*66)
        print('Input data:')
        print(raw_data.description)
        print('='*66)
        print('Transformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('='*66)
        print('Patch size:')
        print(" x ".join(str(p) for p in patch_size))
        print('=' * 66)

    sys.stdout.flush()

    ## sample patches from each pair of transformed raw images
    X = np.empty((n_patches,)+tuple(patch_size),dtype=np.float32)
    Y = np.empty_like(X)

    for i, (x,y,_axes,mask) in tqdm(enumerate(image_pairs),total=n_images,disable=(not verbose)):
        if i >= n_images:
            warnings.warn('more raw images (or transformations thereof) than expected, skipping excess images.')
            break
        if i==0:
            axes = axes_check_and_normalize(_axes,len(patch_size))
            channel = axes_dict(axes)['C']
        # checks
        # len(axes) >= x.ndim or _raise(ValueError())
        axes == axes_check_and_normalize(_axes) or _raise(ValueError('not all images have the same axes.'))
        x.shape == y.shape or _raise(ValueError())
        mask is None or mask.shape == x.shape or _raise(ValueError())
        (channel is None or (isinstance(channel,int) and 0<=channel<x.ndim)) or _raise(ValueError())
        channel is None or patch_size[channel]==x.shape[channel] or _raise(ValueError('extracted patches must contain all channels.'))

        _Y,_X = sample_patches_from_multiple_stacks((y,x), patch_size, n_patches_per_image, mask, patch_filter)

        s = slice(i*n_patches_per_image,(i+1)*n_patches_per_image)
        X[s], Y[s] = normalization(_X,_Y, x,y,mask,channel)

    if shuffle:
        shuffle_inplace(X,Y)

    axes = 'SC'+axes.replace('C','')
    if channel is None:
        X = np.expand_dims(X,1)
        Y = np.expand_dims(Y,1)
    else:
        X = np.moveaxis(X, 1+channel, 1)
        Y = np.moveaxis(Y, 1+channel, 1)

    if save_file is not None:
        print('Saving data to %s.' % str(Path(save_file)))
        save_training_data(save_file, X, Y, axes)

    return X,Y,axes


def create_patches_reduced_target(
        raw_data,
        patch_size,
        n_patches_per_image,
        reduction_axes,
        target_axes = None, # TODO: this should rather be part of RawData and also exposed to transforms
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
    reduction_axes = axes_check_and_normalize(reduction_axes,disallowed='S')

    transforms = kwargs.get('transforms')
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    transforms.insert(0,broadcast_target(target_axes))
    kwargs['transforms'] = transforms

    save_file = kwargs.pop('save_file',None)

    if any(s is None for s in patch_size):
        patch_axes = kwargs.get('patch_axes')
        if patch_axes is not None:
            _transforms = list(transforms)
            _transforms.append(permute_axes(patch_axes))
        else:
            _transforms = transforms
        tf = Transform(*zip(*_transforms))
        image_pairs = compose(*tf.generator)(raw_data.generator())
        x,y,axes,mask = next(image_pairs) # get the first entry from the generator
        patch_size = list(patch_size)
        for i,(a,s) in enumerate(zip(axes,patch_size)):
            if s is not None: continue
            a in reduction_axes or _raise(ValueError("entry of patch_size is None for non reduction axis %s." % a))
            patch_size[i] = x.shape[i]
        patch_size = tuple(patch_size)
        del x,y,axes,mask

    X,Y,axes = create_patches (
        raw_data            = raw_data,
        patch_size          = patch_size,
        n_patches_per_image = n_patches_per_image,
        **kwargs
    )

    ax = axes_dict(axes)
    for a in reduction_axes:
        a in axes or _raise(ValueError("reduction axis %d not present in extracted patches" % a))
        n_dims = Y.shape[ax[a]]
        if n_dims == 1:
            warnings.warn("extracted target patches already have dimensionality 1 along reduction axis %s." % a)
        else:
            t = np.take(Y,(1,),axis=ax[a])
            Y = np.take(Y,(0,),axis=ax[a])
            i = np.random.choice(Y.size,size=100)
            if not np.all(t.flat[i]==Y.flat[i]):
                warnings.warn("extracted target patches vary along reduction axis %s." % a)

    if save_file is not None:
        print('Saving data to %s.' % str(Path(save_file)))
        save_training_data(save_file, X, Y, axes)

    return X,Y,axes


# Misc

def shuffle_inplace(*arrs,**kwargs):
    seed = kwargs.pop('seed', None)
    if seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(seed=seed)
    state = rng.get_state()
    for a in arrs:
        rng.set_state(state)
        rng.shuffle(a)


########################################## MITOCHONDRIA PART ##########################################

def sample_patches_in_image(datas, patch_size, n_sample_size, cpt, datas_mask=None, patch_filter=None, verbose=False):
    """ Sample matching patches of size `patch_size` from all arrays in `datas` """

    # TODO: checks

    if patch_filter is None:
        patch_mask = np.ones(datas[0].shape,dtype=np.bool)
    else:
        patch_mask = patch_filter(datas, patch_size)

    if datas_mask is not None:
        # TODO: Test this
        warnings.warn('Using pixel masks for raw/transformed images not tested.')
        datas_mask.shape == datas[0].shape or _raise(ValueError())
        datas_mask.dtype == np.bool or _raise(ValueError())
        from scipy.ndimage.filters import minimum_filter
        patch_mask &= minimum_filter(datas_mask, patch_size, mode='constant', cval=False)

    # Create patches
    y, x = datas
    patches_x = []
    patches_y = []
    n_height, n_width = n_sample_size
    for i in range (n_height):
        for j in range (n_width):
            end_height = i * patch_size[0] + patch_size[0]
            end_width = j * patch_size[1] + patch_size[1]

            # We are in the bottom right corner of the image
            if end_height > x.shape[0] and end_width > x.shape[1]:
                patch_x, patch_y = x[x.shape[0] - patch_size[0]:, x.shape[1] - patch_size[1]:], y[y.shape[0] - patch_size[1]:, y.shape[1] - patch_size[1]:]

            # We are on the bottom of the image
            elif end_height > x.shape[0]:
                start_width = j * patch_size[1]
                end_width = j * patch_size[1] + patch_size[1]
                patch_x, patch_y = x[x.shape[0] - patch_size[0]:, start_width:end_width], y[y.shape[0] - patch_size[0]:, start_width:end_width]

            # We are on the right of the image
            elif end_width > x.shape[1]:
                start_height = i * patch_size[0]
                end_height = i * patch_size[0] + patch_size[0]
                patch_x, patch_y = x[start_height:end_height, x.shape[1] - patch_size[1]:], y[start_height:end_height, y.shape[1] - patch_size[1]:]

            # We do not touch an end corner of the image
            else :
                start_height = i * patch_size[0]
                end_height = i * patch_size[0] + patch_size[0]
                start_width = j * patch_size[1]
                end_width = j * patch_size[1] + patch_size[1]
                patch_x, patch_y = x[start_height:end_height, start_width:end_width], y[start_height:end_height, start_width:end_width]

            # Check whether patch is valid or not : contains enough relevant information for training
            # Delete or add it
            patch_grey = cv2.cvtColor(patch_x, cv2.COLOR_RGB2BGR)
            if patch_is_valid(patch_grey, cpt):
                pass
            patches_x.append(patch_x.tolist())
            patches_y.append(patch_y.tolist())

            cpt += 1

    return np.array(patches_y), np.array(patches_x), cpt


def create_patches_mito(
        raw_data,
        patch_size,
        data_path,
        patch_axes = None,
        transforms    = None,
        patch_filter  = no_background_patches(),
        save_file     = None,
        normalization = norm_percentiles(),
        shuffle       = False,
        verbose       = True,
        compo = False,
    ):
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
    compo : bool
        Declare if we compose transformation or if we do multiples

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

    """
    ## images and transforms
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
    tf = Transform(*zip(*transforms))  # convert list of Transforms into Transform of lists
    #print("tf  :", tf)
    #print("tf.generator  :", tf.generator)
    #image_pairs = compose(*tf.generator)(image_pairs)  # combine all transformations with raw images as input
    generators = [tf.generator[i](image_pairs) for i in range (len(tf.generator))] # combine all transformations with raw images as input
    #print("generators  :", generators)
    image_pairs = chain(*generators)
    all_files_names = sorted(os.listdir(data_path + '/train/GT'))

    # Summary
    if verbose:
        print('=' * 66)
        print('Input data:')
        print(raw_data.description)
        print('=' * 66)    # Calculate number of final images
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
    print('len de image_pair_iter : ', len(image_pair_iter))
    for iter, (x, y, _axes, mask) in image_pair_iter:

        # TODO : Tester si les transfo marchent
        #cv2.imwrite('todelete/imageTransform_'+ str(iter) + '.tif', x)

        if x.shape[0]>=patch_size[0] and x.shape[1]>=patch_size[1]:
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
    create_patch_dir('patches')
    create_histo_dir('histos')

    # Create patches for each image
    occupied = 0
    cpt = 0
    print("Building data patches...")
    for i, (x, y, _axes, mask) in tqdm(list_image_pair, total=len(list_image_pair), disable=(not verbose)):
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
        if x.shape[0]>=patch_size[0] and x.shape[1]>=patch_size[1]:
            # Calculate number of patcher per image
            n_patches_height = math.ceil(x.shape[0] / patch_size[0])
            n_patches_width = math.ceil(x.shape[1] / patch_size[1])

            # Create patches
            _Y, _X, cpt = sample_patches_in_image((y, x), patch_size, (n_patches_height, n_patches_width), cpt, mask, patch_filter)
            # Calculate the number of patches per image
            n_patches_per_image = len(_X)

            # If at least on patch is kept within the image
            if n_patches_per_image != 0:

                # Fulfill final container of images with patches
                s = slice(occupied, occupied + n_patches_per_image)
                X[s], Y[s] = normalization(_X, _Y, x, y, mask, channel)

                # Save image if one patch at least is deleted (to check selection mechanism)
                n_patches_height = math.ceil(x.shape[0] / patch_size[0])
                n_patches_width = math.ceil(x.shape[1] / patch_size[1])
                max_patches_per_image = n_patches_height * n_patches_width

                # Saving patches in folder patches
                for k in range (n_patches_per_image):
                    x, y = X[occupied + k], Y[occupied + k]
                    datas = (x, y)
                    try:
                        patch_file_name = all_files_names[i].split('.')[0] + '_' + str(k) + '.STED.ome.tif'
                    except:
                        patch_file_name = all_files_names[i].split('.')[0] + '_' + str(k) + '.tif'
                    save_patch(patch_file_name, datas)

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

    print(f"There are {len(X)} final images for training.")
    return X, Y, axes


def patch_is_valid(patch, patch_nb):
    """ Check whether a patch contains too much noise, or not enough relevant information, which could bias the training
    Parameters
    ----------
        patch : a numpy array image
    Returns
    -------
        True if the patch is kept for training, False otherwise
    """
    # Calculate histogram of saturation channel
    s = cv2.calcHist([patch], [1], None, [256], [0, 256])

    if 0<patch_nb<100:
        plt.figure(patch_nb)
        plt.imshow(patch)
        plt.savefig("todelete/patch_" + str(patch_nb) + ".png")
        #cv2.imwrite("todelete/patch_" + str(patch_nb) + ".tif", patch)
        plt.figure(patch_nb+1000)
        plt.plot(s)
        plt.savefig("histos/plot_" + str(patch_nb) + ".png")
        if patch_nb == 9:
            cv2.imwrite("todelete/patch9.tif", patch)

    # Calculate attribute of the histogram
    pixel_values = np.arange(0, 256)
    mean_histo = np.sum(s.T * pixel_values) / np.sum(s)
    max_histo = np.max(s, axis=0)
    qty_high = np.sum(s[200:])

    #print(f"mean histo : {patch_nb} : {mean_histo}")
    #print(f"max_histo : {patch_nb} : {max_histo}")
    #print(f"qty_high : {patch_nb} : {qty_high}")

    if (mean_histo < 145 and max_histo > 1000 and qty_high < 210) or qty_high == 0.0 or mean_histo>240:
        if save_deleted_patches:
            cv2.imwrite('todelete/deletedPatch_' + str(patch_nb) + '.png', patch)
        return False

    return True

def patch_is_valid_occupation(patch, patch_nb, tshd_noise=25, thshd_occup=0.1):
    """ Check whether a patch contains too much noise, or not enough relevant information, which could bias the training
    Parameters
    ----------
        patch : a numpy array image
    Returns
    -------
        True if the patch is kept for training, False otherwise
    """
    # Calculate histogram of saturation channel
    s = cv2.calcHist([patch], [1], None, [256], [0, 256])

    if 0<patch_nb<100:
        plt.figure(patch_nb)
        plt.imshow(patch)
        plt.savefig("todelete/patch_" + str(patch_nb) + ".png")
        #cv2.imwrite("todelete/patch_" + str(patch_nb) + ".tif", patch)
        plt.figure(patch_nb+1000)
        plt.plot(s)
        plt.savefig("histos/plot_" + str(patch_nb) + ".png")
        if patch_nb == 9:
            cv2.imwrite("todelete/patch9.tif", patch)

    # Calculate percentage of occupation
    total_pixel = np.sum(s)
    occupation = np.sum(s[tshd_noise:])
    occupation_min = thshd_occup * total_pixel

    # Delete if occupation is not enough
    if occupation < occupation_min:
        return False

    return True



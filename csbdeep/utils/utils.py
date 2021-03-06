from __future__ import print_function, unicode_literals, absolute_import, division

import os
import numpy as np
import json
import collections
import platform
import random
from six.moves import range, map, reduce
from .six import Path
import matplotlib.pyplot as plt
from zipfile import ZipFile
import shutil
import cv2


def is_tf_backend():
    from .tf import keras_import
    K = keras_import('backend')
    return K.backend() == 'tensorflow'


def backend_channels_last():
    from .tf import keras_import
    K = keras_import('backend')
    assert K.image_data_format() in ('channels_first', 'channels_last')
    return K.image_data_format() == 'channels_last'


def move_channel_for_backend(X, channel):
    if backend_channels_last():
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel, 1)


###


def load_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)


def save_json(data, fpath, **kwargs):
    with open(fpath, 'w') as f:
        f.write(json.dumps(data, **kwargs))


###


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)
    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)
    return x


def normalize_minmse(x, target):
    """Affine rescaling of x, such that the mean squared error to target is minimal."""
    cov = np.cov(x.flatten(), target.flatten())
    alpha = cov[0, 1] / (cov[0, 0] + 1e-10)
    beta = target.mean() - alpha * x.mean()
    return alpha * x + beta


###

def _raise(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)


def compose(*funcs):
    return lambda x: reduce(lambda f, g: g(f), funcs, x)


###


def download_and_extract_zip_file(url, targetdir='.', verbose=True):
    import csv
    from six.moves.urllib.request import urlretrieve
    from six.moves.urllib.parse import urlparse
    from zipfile import ZipFile

    res = urlparse(url)
    if res.scheme in ('', 'file'):
        url = Path(res.path).resolve().as_uri()
        # local file, 'urlretrieve' will not make a copy
        # -> don't delete 'downloaded' file
        delete = False
    else:
        delete = True

    # verbosity levels:
    # - 0: no messages
    # - 1: status messages
    # - 2: status messages and list of all files
    if isinstance(verbose, bool):
        verbose *= 2

    log = (print) if verbose else (lambda *a, **k: None)

    targetdir = Path(targetdir)
    if not targetdir.is_dir():
        targetdir.mkdir(parents=True, exist_ok=True)

    provided = []

    def content_is_missing():
        try:
            filepath, http_msg = urlretrieve(url + '.contents')
            with open(filepath, 'r') as contents_file:
                contents = list(csv.reader(contents_file, delimiter='\t'))
        except:
            return True
        finally:
            if delete:
                try:
                    os.unlink(filepath)
                except:
                    pass

        for size, relpath in contents:
            size, relpath = int(size.strip()), relpath.strip()
            entry = targetdir / relpath
            if not entry.exists():
                return True
            if entry.is_dir():
                if not relpath.endswith('/'): return True
            elif entry.is_file():
                if relpath.endswith('/') or entry.stat().st_size != size: return True
            else:
                return True
            provided.append(relpath)

        return False

    if content_is_missing():
        try:
            log('Files missing, downloading...', end='')
            filepath, http_msg = urlretrieve(url)
            with ZipFile(filepath, 'r') as zip_file:
                log(' extracting...', end='')
                zip_file.extractall(str(targetdir))
                provided = zip_file.namelist()
            log(' done.')
        finally:
            if delete:
                try:
                    os.unlink(filepath)
                except:
                    pass
    else:
        log('Files found, nothing to download.')

    if verbose > 1:
        log('\n' + str(targetdir) + ':')
        consume(map(lambda x: log('-', Path(x)), provided))


############################################# MITOCHONDRIES FILES #####################################################

def extract_zip_file(folder_path, targetdir='data_mito', zip=False):
    if zip:
        try:
            log('Files missing, extracting...', end='')
            with ZipFile(folder_path, 'r') as zip:
                zip.extractall(str(targetdir))
                print(' done.')
        except:
            shutil.rmtree(targetdir)
            with ZipFile(folder_path, 'r') as zip:
                zip.extractall(str(targetdir))
                print(' done.')
    else:
        try:
            shutil.copytree(folder_path, targetdir)
        except:
            shutil.rmtree(targetdir)
            shutil.copytree(folder_path, targetdir)


###


def axes_check_and_normalize(axes, length=None, disallowed=None, return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes is not None or _raise(ValueError('axis cannot be None.'))
    axes = str(axes).upper()
    consume(
        a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s." % (a, list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'." % a)) for a in axes)
    consume(axes.count(a) == 1 or _raise(ValueError("axis '%s' occurs more than once." % a)) for a in axes)
    length is None or len(axes) == length or _raise(ValueError('axes (%s) must be of length %d.' % (axes, length)))
    return (axes, allowed) if return_allowed else axes


def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes, return_allowed=True)
    return {a: None if axes.find(a) == -1 else axes.find(a) for a in allowed}
    # return collections.namedtuple('Axes',list(allowed))(*[None if axes.find(a) == -1 else axes.find(a) for a in allowed ])


def move_image_axes(x, fr, to, adjust_singletons=False):
    """
    x: ndarray
    fr,to: axes string (see `axes_dict`)
    """
    fr = axes_check_and_normalize(fr, length=x.ndim)
    to = axes_check_and_normalize(to)

    fr_initial = fr
    x_shape_initial = x.shape
    adjust_singletons = bool(adjust_singletons)
    if adjust_singletons:
        # remove axes not present in 'to'
        slices = [slice(None) for _ in x.shape]
        for i, a in enumerate(fr):
            if (a not in to) and (x.shape[i] == 1):
                # remove singleton axis
                slices[i] = 0
                fr = fr.replace(a, '')
        x = x[tuple(slices)]
        # add dummy axes present in 'to'
        for i, a in enumerate(to):
            if (a not in fr):
                # add singleton axis
                x = np.expand_dims(x, -1)
                fr += a

    if set(fr) != set(to):
        _adjusted = '(adjusted to %s and %s) ' % (x.shape, fr) if adjust_singletons else ''
        raise ValueError(
            'image with shape %s and axes %s %snot compatible with target axes %s.'
            % (x_shape_initial, fr_initial, _adjusted, to)
        )

    ax_from, ax_to = axes_dict(fr), axes_dict(to)
    if fr == to:
        return x
    return np.moveaxis(x, [ax_from[a] for a in fr], [ax_to[a] for a in fr])


###


def choice(population, k=1, replace=True):
    ver = platform.sys.version_info
    if replace and (ver.major, ver.minor) in [(2, 7), (3, 5)]:  # python 2.7 or 3.5
        # slow if population is large and not a np.ndarray
        return list(np.random.choice(population, k, replace=replace))
    else:
        try:
            # save state of 'random' and set seed using 'np.random'
            state = random.getstate()
            random.seed(np.random.randint(np.iinfo(int).min, np.iinfo(int).max))
            if replace:
                # sample with replacement
                return random.choices(population, k=k)
            else:
                # sample without replacement
                return random.sample(population, k=k)
        finally:
            # restore state of 'random'
            random.setstate(state)


def save_figure(moment, datatype, xaxis=None, yaxis=None):
    if not os.path.isdir('fig/'):
        os.makedirs('fig/')
    if not os.path.isdir('fig/' + moment + '/'):
        os.makedirs('fig/' + moment + '/')

    file_name = 'fig/' + moment + '/' + datatype + '.jpg'
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.savefig(file_name, bbox_inches='tight')


def save_patch(save_name, datas):
    x, y = datas

    path_save_x = 'savings/patches/train/low/' + save_name
    cv2.imwrite(path_save_x, x)

    path_save_y = 'savings/patches/train/GT/' + save_name
    cv2.imwrite(path_save_y, y)


def create_patch_dir(patch_dir_name):
    try:
        os.makedirs(patch_dir_name + '/')
        os.makedirs(patch_dir_name + '/train/')
        os.makedirs(patch_dir_name + '/train/GT/')
        os.makedirs(patch_dir_name + '/train/low')
    except:
        shutil.rmtree(patch_dir_name)
        os.makedirs(patch_dir_name + '/')
        os.makedirs(patch_dir_name + '/train/')
        os.makedirs(patch_dir_name + '/train/GT/')
        os.makedirs(patch_dir_name + '/train/low')


def create_dir(dir_name):
    try:
        os.makedirs(dir_name + '/')
    except:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name + '/')


def normalize_0_255(datas):
    normalized = []
    for i in range(len(datas)):
        norm = datas[i] * 255 / datas[i].max()
        normalized.append(norm)
    return normalized


def vrange(starts, lengths):
    """Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:

        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    lengths = np.array(lengths)
    starts = np.array(starts)
    stops = np.asarray(lengths+starts)
    return np.repeat(stops - lengths.cumsum(), lengths) + np.arange(lengths.sum())


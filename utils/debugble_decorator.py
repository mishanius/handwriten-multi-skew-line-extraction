import logging
import os
import time
from functools import wraps
from os import listdir
import numpy as np

import matplotlib.pyplot as plt

from utils.MetricLogger import MetricLogger, Singleton

DEFAULT_CACHE_PATH = os.path.join(os.path.abspath(os.getcwd()), "numpy_cache")
GLOBAL_VERSION_MAPPING = {}


class CacheSwitch(metaclass=Singleton):
    def __init__(self):
        super().__init__()
        self.value = True

class PartialImageSwitch(metaclass=Singleton):
    def __init__(self):
        super().__init__()
        self.value = True


def timed(lgnm=None, agregated=False, log_max_runtime=False, verbose=False):
    def inner_function(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = MetricLogger()
            name = func.__name__ if lgnm is None else lgnm
            if verbose:
                logger.info("started {}".format(name))
            t_start = time.time()
            result = func(*args, **kwargs)
            t_end = time.time()

            if agregated:
                logger.inc_metric(name, (t_end - t_start) * 1000)
            if log_max_runtime:
                logger.log_max(name, (t_end - t_start) * 1000)
            if verbose:
                logger.info("finished {} took :{}(ms)".format(name, (t_end - t_start) * 1000))
            return result

        return wrapper

    return inner_function


def partial_image(index_of_output, name, binarize_image=True):
    def inner_function(func):
        @wraps(func)
        def wrapper(*args):
            next_version = GLOBAL_VERSION_MAPPING.get(name, 1)
            cm = plt.get_cmap('gray')
            kw = {'cmap': cm, 'interpolation': 'none', 'origin': 'upper'}
            result = func(*args)
            if PartialImageSwitch().value:
                temp = result[index_of_output] > 0 if binarize_image else result
                if index_of_output < 0:
                    plt.imshow(temp, **kw)
                else:
                    plt.imshow(temp, **kw)
                plt.title('partial result: %s %d' % (name, next_version))
                plt.show()
            return result

        return wrapper

    return inner_function


def numpy_cached(func):
    def wrapper(*args, **kwargs):
        if not os.path.exists(DEFAULT_CACHE_PATH):
            os.mkdir(DEFAULT_CACHE_PATH)
        version_to_find = GLOBAL_VERSION_MAPPING.get(func.__name__, 1)
        GLOBAL_VERSION_MAPPING[func.__name__] = version_to_find + 1
        if CacheSwitch().value:
            for f in listdir(DEFAULT_CACHE_PATH):
                if f == "{}_{}.npz".format(func.__name__, version_to_find):
                    MetricLogger().info("%s found cached file" % func.__name__)
                    npzfile = np.load(os.path.join(DEFAULT_CACHE_PATH, f))
                    return unpack_npz(npzfile)

        result, *_ = func(*args, **kwargs), None
        if CacheSwitch().value:
            cache = {}
            if type(result) == tuple:
                for idx, r in enumerate(result):
                    cache = pack_value(cache, r, idx)
            else:
                cache = pack_value(cache, result, 0)
            np.savez(os.path.join(DEFAULT_CACHE_PATH, "%s_%d" % (func.__name__, version_to_find)), **cache)
        return result

    return wrapper


def pack_value(cache, arr, idx):
    log = logging.getLogger('basic_metric')
    if type(arr).__module__ == np.__name__:
        cache['%d_np' % idx] = arr
    elif type(arr) == int:
        cache['%d_int' % idx] = np.array(arr)
    elif type(arr) == list:
        cache['%d_list' % idx] = np.array(arr)
    else:
        log.debug("unknown type caching problem")
    return cache


def unpack_npz(npz_file):
    vals = []
    for f in npz_file.files:
        modif_type = f.split('_')[1]
        if modif_type == 'np':
            r = npz_file[f]
        elif modif_type == 'int':
            r = int(npz_file[f])
        elif modif_type == 'list':
            r = npz_file[f].tolist()
        else:
            raise RuntimeError("unknown modif_type")
        if len(npz_file.files) == 1:
            return r
        vals.append(r)
    return vals

import os
import sys
from sklearn.model_selection import KFold

try:
    # In case we're in the tensorflow environment
    import lightgbm as lgb
except:
    pass

from dstools.ds_statsml._alt_metrics import *
from time import time


def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        with open(os.devnull, "w") as devNull:
            original = sys.stdout
            sys.stdout = devNull    # suppress printing
            func(*args, **kwargs)
            sys.stdout = original   # re-enable printing

    return func_wrapper


def depth(d, level=1):
    '''
    Get the depth of a dictionary
    :param d: The dictionary
    '''
    if not isinstance(d, dict) or not d:
        return level
    return max(depth(d[k], level + 1) for k in d)


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms'.format(method.__name__, (te - ts) * 1000))
        return result
    return timed


def get_array_batches(a, max_batch_size=4):
    a = np.array(a)
    cv = KFold(n_splits=len(a) // (max_batch_size - 1))

    return [list(a[x[1]]) for x in cv.split(a)]

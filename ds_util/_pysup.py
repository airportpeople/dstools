import os
import sys
from sklearn.model_selection import KFold
from collections import Counter
from copy import deepcopy

try:
    # In case we're in the tensorflow environment
    import lightgbm as lgb
except:
    pass

from ds_statsml._alt_metrics import *
from time import time


def blockPrinting(func):
    """
    *Decorator* to block printing of anything within the defined function. Handy for defining functions where you know several sub-functions print
    a whole lot of junk.

    For example, if you have:

        def printfunc():
            print('blah')

        @blockPrinting
        def outerfunc():
            printfunc()
            print('outer blah')

        >>> outerfunc()
        [Out 1]:

    You get nothing.
    """
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        with open(os.devnull, "w") as devNull:
            original = sys.stdout
            sys.stdout = devNull    # suppress printing
            func(*args, **kwargs)
            sys.stdout = original   # re-enable printing

    return func_wrapper


def depth(d, level=1):
    """
    Get the maximum depth of a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to determine depth
    level : recursive variable
        LEAVE EQUAL TO 1 (recursive function)

    Returns
    -------
    (int) The maximum depth of the dictionary (i.e., the most number of nested dictionaries).
    """
    if not isinstance(d, dict) or not d:
        return level
    return max(depth(d[k], level + 1) for k in d)


def flatten_list(l):
    '''
    Convert a list of lists (two-dimensional shape) to a single list.

    Parameters
    ----------
    l : list-like
        The original list

    Returns
    -------
    (list) A flattened version of the original list of lists.
    '''
    return [item for sublist in l for item in sublist]


def timeit(method):
    """
    *Decorator* function to log the time to execute the function defined each time it's called.
    """
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f'{method.__name__}  {round((te - ts) * 1000, 3)} ms')
        return result
    return timed


def get_array_batches(a, max_batch_size=4):
    """
    Given an array, break it up into batches of `max_batch_size` or less. In general, the batches should be of size `max_batch_size`. But,
    if this number doesn't divide evenly into `len(a)`, then the batches that are not of size `max_batch_size` will be `max_batch_size - 1`.

    Parameters
    ----------
    a : array-like
        The array to be divided into batches.
    max_batch_size : int, optional
        The maximum size for each batch

    Returns
    -------
    (list of lists) A list containing batches (each in the form of a list).
    """
    a = np.array(a)
    cv = KFold(n_splits=int(np.ceil(len(a) / max_batch_size)))

    return [list(a[x[1]]) for x in cv.split(a)]


def numerate_dupes(x):
    e_counts = dict(Counter(x))
    ooms = {e: int(np.floor(np.log10(e_counts[e]))) for e in x}
    x_num = deepcopy(x)
    duplicates = {k: 0 for k in x}

    for i, e in enumerate(x):
        if e_counts[e] > 1:
            duplicates[e] += 1
            x_num[i] = e + str(duplicates[e]).zfill(ooms[e] + 1)

    return x_num
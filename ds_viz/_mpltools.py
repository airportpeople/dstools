import matplotlib.pyplot as plt
import numpy as np


def cleanup_xaxis(axis, stepsize, rotation):
    start, end = axis.get_xlim()
    axis.xaxis.set_ticks(np.arange(start, end, stepsize))

    for tick in axis.get_xticklabels():
        tick.set_rotation(rotation)

    return axis
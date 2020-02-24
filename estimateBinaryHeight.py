from scipy.ndimage import label as bwlabel
from skimage.measure import regionprops
import numpy as np
import statistics

from skimage.morphology import reconstruction


def estimateBinaryHeight(binary, margins, ths_hight=200, ths_low=10):
    cleanBin = clear_margins(binary, margins)
    L, _ = bwlabel(np.transpose(cleanBin), np.ones((3,3)))
    L = np.transpose(L)
    props = regionprops(L)
    height = []
    for prop in props:
        h = prop.bbox[2] - prop.bbox[0]
        if ths_low <= h <= ths_hight:
            height.append(prop.bbox[2] - prop.bbox[0])

    mu = statistics.mean(height)
    sigma = statistics.stdev(height)
    lower = mu / 2
    upper = (mu + sigma / 2) / 2

    return lower, upper


def clear_margins(binary, margins):
    if margins == 0:
        return binary
    else:
        L, _ = bwlabel(binary, np.ones((3,3)))
        row, col = binary.shape
        central_row = int(np.floor(row * margins))
        central_col = int(np.floor(col * margins))
        mask = np.full((row, col), False)
        mask[central_row: -central_row, central_col: -central_col] = True
        return reconstruction(np.logical_and(L > 0, mask), L > 0)

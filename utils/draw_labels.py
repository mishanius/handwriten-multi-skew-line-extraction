import numpy as np
import cv2


def draw_labels(L, Labels):
    L = np.uint16(L)
    LUT = np.zeros(65536, np.uint16)
    LUT[1:len(Labels) + 1] = Labels
    result = np.double(LUT[L])
    return result


def gen_lut():
    """
    Generate a label colormap compatible with opencv lookup table, based on
    Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
    appendix C2 `Pseudocolor Generation`.
    :Returns:
      color_lut : opencv compatible color lookup table
    """
    tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
    arr = np.arange(256)
    r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
    g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
    b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
    return np.concatenate([[[r]], [[g]], [[b]]]).T


def labels2rgb(labels):
    """
    Convert a label image to an rgb image using a lookup table
    :Parameters:
      labels : an image of type np.uint8 2D array
      lut : a lookup table of shape (256, 3) and type np.uint8
    :Returns:
      colorized_labels : a colorized label image
    """
    lut = gen_lut()
    return cv2.LUT(cv2.merge((labels, labels, labels)), lut)

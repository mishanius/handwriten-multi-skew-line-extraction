import numpy as np

def draw_labels(L, Labels):
    L = np.uint16(L)
    size = len(L.flatten())
    LUT = np.zeros(size, np.uint16)
    LUT[1:len(Labels) + 1] = Labels
    result = np.double(LUT[L])
    return result

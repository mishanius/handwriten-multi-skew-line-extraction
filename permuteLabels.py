import numpy as np


def permuteLabels(lines):
    lines = lines.astype(np.uint16)
    unique_labels = np.delete(np.unique(np.array(lines)), 0)
    p = np.random.permutation(range(1, len(unique_labels) + 1))
    LUT = np.zeros((65536), dtype=np.uint16)

    for i in range(len(unique_labels)):
        LUT[unique_labels[i]] = p[i]

    new_lines = np.array(LUT[lines], dtype=np.double)
    return [new_lines, len(unique_labels)]

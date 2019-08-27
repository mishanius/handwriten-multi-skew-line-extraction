import numpy as np
def permuteLabels(Lines):
    Lines = Lines.astype(np.uint16)
    uniqueLabels = np.unique(np.array(Lines))
    np.delete(uniqueLabels, 0)
    p = np.random.permutation(len(uniqueLabels))
    LUT = np.zeros((65536), dtype=np.uint16)
    newLinesNum = len(uniqueLabels)

    for i in range(newLinesNum):
        LUT[uniqueLabels[i]] = p[i]

    newLines = np.array(LUT[Lines], dtype=np.double)
    return [newLines, newLinesNum]
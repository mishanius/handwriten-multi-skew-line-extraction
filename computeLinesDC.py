from sklearn.neighbors import NearestNeighbors
import numpy as np
from extractCentroids import extractCentroids
from skimage.measure import regionprops
def computeLinesDC(Lines, numLines,L, num, upperHeight):
    X = extractCentroids(L);
    Lines = Lines.astype(np.int)
    temp = regionprops(Lines)
    Dc = np.zeros((numLines+1, num))

    for i in range(numLines - 1):
        pixelList = temp[i].coords
        print("fitting neighbours")
        nbrs = NearestNeighbors(1).fit(pixelList)
        print("Done fitting neighbours")
        D, _ = nbrs.kneighbors(X)

        if len(pixelList) == 0:
            Dc[i] = np.inf
        else:
            Dc[i] = D.transpose()

    Dc[numLines] = 5 * upperHeight
    return Dc
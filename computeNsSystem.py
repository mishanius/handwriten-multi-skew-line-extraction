from extractCentroids import extractCentroids
from sklearn.neighbors import NearestNeighbors
import numpy as np
def computeNsSystem(L, num):
    X = extractCentroids(L)
    K = 2
    nbrs = NearestNeighbors(K+1).fit(X)
    distances, indexes = nbrs.kneighbors(X)
    # remove first column
    indexes = indexes[:, 1:]
    distances = distances[:, 1:]

    sparseNs = np.zeros((num, num))

    for j in range(K):
        for i in range(num):
            sparseNs[i, indexes[i, j]] = distances[i, j]
    temp = sparseNs - sparseNs.transpose()
    temp[temp > 0] = 0
    temp = np.abs(temp)+ sparseNs
    sparseNs = np.triu(temp)
    return sparseNs

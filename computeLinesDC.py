from sklearn.neighbors import NearestNeighbors
import numpy as np
from extractCentroids import extractCentroids
from skimage.measure import regionprops


def computeLinesDC(Lines, numLines, L, num, upperHeight):
    X = extractCentroids(L)
    Lines = Lines.astype(np.int)
    temp = regionprops(Lines)
    Dc = np.zeros((numLines + 1, num))

    for i in range(numLines):
        pixelList = temp[i].coords
        nbrs = NearestNeighbors(1).fit(pixelList)
        D, _ = nbrs.kneighbors(X)

        if len(pixelList) == 0:
            Dc[i] = np.inf
        else:
            Dc[i] = D.transpose()

    Dc[numLines] = 5 * upperHeight
    return Dc


def compute_lines_data_cost(line_mask_lables, num_labeled_lines, raw_components, num, upper_height):
    raw_components_centroids = [[prop.centroid[1], prop.centroid[0]] for prop in regionprops(raw_components)]
    line_mask_lables = line_mask_lables.astype(np.int)
    data_cost = np.zeros((num_labeled_lines+1, num))
    for index, temp in enumerate(regionprops(np.transpose(line_mask_lables))):
        pixel_list = temp.coords
        if len(pixel_list) == 0:
            data_cost[index] = np.inf
        else:
            nbrs = NearestNeighbors(1).fit(pixel_list)
            data_cost[index] = nbrs.kneighbors(raw_components_centroids)[0].transpose()
    data_cost[num_labeled_lines] = 5 * upper_height
    return data_cost.transpose()

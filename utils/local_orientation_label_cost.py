import math

from skimage.measure import regionprops
from plantcv.plantcv import dilate, erode
import numpy as np
from sklearn.decomposition import PCA


def local_orientation_label_cost(labeled_lines, labeled_lines_num, intact_lines_num, max_orientation,
                                 max_response, theta, radius_constant=18):
    pixel_list = regionprops(np.transpose(labeled_lines))
    label_cost = np.zeros((labeled_lines_num + 1, 1))
    local_max_orientation = np.zeros((labeled_lines_num, 1))
    logical = labeled_lines > 0
    logical_double = logical.astype(np.double)
    # TODO decide number of iterations
    border_mask = np.logical_and(logical, np.logical_not(erode(logical_double, 3, 1)))
    border_mask = border_mask.astype(np.double)
    # area divide by perimeter
    sw = np.sum(logical_double) / np.sum(border_mask)
    se = round(sw * radius_constant)
    line_theta = np.zeros((labeled_lines_num, 1))
    for i in range(intact_lines_num, labeled_lines_num):
        x = pixel_list[i].coords
        try:
            pca = PCA()
            pca_res = pca.fit(x)
            pcav = pca_res.components_[0]
            line_theta[i] = math.atan(pcav[1] / pcav[0])
        except Exception as e:
            line_theta[i] = np.inf
            continue
        logical = labeled_lines == i+1
        logical_double = logical.astype(np.double)
        #TODO number of iterations
        mask = dilate(logical_double, se, 1)
        res = estimate_local_orientations(max_orientation, max_response, theta, mask)
        index = np.argmax(res[:, 1])
        local_max_orientation[i] = res[index, 1]
        label_cost[i] = 10 * np.exp(50 * (1 - abs(np.cos(math.radians(local_max_orientation[i]) - line_theta[i]))))
    return label_cost


def estimate_local_orientations(max_orientation, max_response, theta, mask):
    res = np.zeros((len(theta), 2))
    flat_img = max_orientation.flatten()
    flat_mask = mask.flatten()
    flat_response = max_response.flatten()
    for i in range(len(flat_img)):
        if flat_mask[i] and flat_img[i]:
            loc = flat_img[i]
            res[loc, 0] = theta[loc]
            res[loc, 1] = res[loc, 1] + flat_response[i]
    return res

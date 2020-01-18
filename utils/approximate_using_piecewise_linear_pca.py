import math

from scipy.interpolate import UnivariateSpline
from skimage.measure import regionprops
import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA


def approximate_using_piecewise_linear_pca(lines, num, marked, ths):
    pca = PCA()
    temp = regionprops(lines)
    max_num_of_knots = 200
    fitting = np.zeros((num, max_num_of_knots - 1))
    for i in range(num):
        if i in marked:
            fitting[i, :] = [0]
            continue

        try:
            pixel_list = temp[i].coords
            pca_res = pca.fit(pixel_list)
            pcav = pca_res.components_[0]
            theta = math.atan(pcav[1] / pcav[0])
            transformation = np.array([[math.cos(-theta), -math.sin(-theta)], [math.sin(-theta), math.cos(-theta)]])
            rotated_pixels = np.matmul(transformation, np.transpose(pixel_list))
        except Exception as e:
            fitting[i, :] = [np.inf]
            continue

        x_rotated = rotated_pixels[0, :]
        y_rotated = rotated_pixels[1, :]
        zipped = sorted(zip(x_rotated, y_rotated))
        sorted_x, sorted_y = list(zip(*zipped))
        slm = UnivariateSpline(sorted_x, sorted_y, s=1, k=1)
        knots = slm.get_knots()
        print("knots:{}".format(len(knots)))
        coeffs = slm.get_coeffs()
        for index in range(len(knots) - 1):
            x_end_point = knots[index:index + 2]
            y_end_point = coeffs[index:index + 2]
            p = np.poly1d(np.polyfit(x_end_point, y_end_point, 1))

            indices = np.argwhere((x_end_point[0] <= x_rotated) & (x_rotated <= x_end_point[1]))
            x_ = x_rotated[indices]
            y_ = y_rotated[indices]
            y_hat = p(x_)
            fitting[i, index] = LA.norm(y_hat - y_, 1) / len(x_)
    fitting = np.max(fitting, axis=1)
    return fitting

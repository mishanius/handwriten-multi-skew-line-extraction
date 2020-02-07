import math

from scipy.interpolate import UnivariateSpline
from skimage.measure import regionprops
import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA


def approximate_using_piecewise_linear_pca(lines, num, marked, ths):
    pca = PCA()
    temp = regionprops(lines)
    fitting = np.zeros((num, 1))
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
        try:
            slm = find_spline_with_numberofknots(sorted_x, sorted_y, 20, threshold=3)
        except Exception as e:
            fitting[i, :] = [0]
            continue
        # slm = UnivariateSpline(sorted_x, sorted_y, len(sorted_x) * 100, k=1)
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
            fitting[i] = max(fitting[i], LA.norm(y_hat - y_, 1) / len(x_))
    return fitting


def find_spline_with_numberofknots(data_x, data_y, desired_number_of_knots, threshold=0, max_iterations=10):
    max = 1000
    min = 0
    iteration = 0
    slm = UnivariateSpline(data_x, data_y, s=len(data_x) * max, k=1)
    number_of_knots = slm.get_knots().size
    #determine max
    while slm.get_knots().size > desired_number_of_knots:
        max = max*2
        slm = UnivariateSpline(data_x, data_y, s=len(data_x) * max, k=1)
    factor = max
    while not (
            abs(desired_number_of_knots - number_of_knots) <= threshold and number_of_knots <= desired_number_of_knots):
        slm = UnivariateSpline(data_x, data_y, s=len(data_x) * factor, k=1)
        number_of_knots = slm.get_knots().size
        if slm.get_knots().size > desired_number_of_knots:
            min = factor
            if max - factor < 0.01:
                max = max * 2
            factor = factor + (max - factor) / 2
        elif slm.get_knots().size < desired_number_of_knots:
            if factor - min < 0.01:
                min = min - 1
            max = factor
            factor = factor - (factor - min) / 2
        iteration += 1
        if iteration>max_iterations:
            print("cant find normal spline current number of knots:{}".format(number_of_knots))
            if desired_number_of_knots < number_of_knots:
                raise RuntimeError("cant find normal spline")
            else:
                return slm
    return slm

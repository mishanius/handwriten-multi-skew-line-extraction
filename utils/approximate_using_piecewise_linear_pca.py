import math
from plantcv.plantcv import dilate
import numpy as np
import matplotlib.pyplot as plt
import pwlf
from scipy.interpolate import UnivariateSpline
from skimage.measure import regionprops
import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA

from utils.debugble_decorator import timed


@timed
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
            # slm, knots = alternativespline_fitting(sorted_x, sorted_y, 3)
            slm = find_spline_with_numberofknots(sorted_x, sorted_y, 20, threshold=3, max_iterations=30)
            knots = slm.get_knots()
        except Exception as e:
            fitting[i, :] = [0]
            continue
        # slm = UnivariateSpline(sorted_x, sorted_y, len(sorted_x) * 100, k=1)
        print("knots:{}".format(len(knots)))
        coeffs = slm.get_coeffs()
        # coeffs = slm.predict(knots)
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


@timed
def alternativespline_fitting(xdata, y, number_of_knots):
    # initialize piecewise linear fit with your x and y data
    my_pwlf = pwlf.PiecewiseLinFit(xdata, y)

    # fit the data for four line segments
    res = my_pwlf.fit(number_of_knots)

    # predict for the determined points
    xHat = np.linspace(min(xdata), max(xdata), num=10000)
    yHat = my_pwlf.predict(xHat)
    print("knots:{}, ys:{}".format(res, my_pwlf.predict(res)))
    # plot the results
    plt.figure()
    plt.plot(xdata, y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.show()
    return my_pwlf, res


def find_spline_with_numberofknots(data_x, data_y, desired_number_of_knots, threshold=0, max_iterations=100):
    max_f = 1000
    min_f = 0
    iteration = 0
    closesed = None
    slm = UnivariateSpline(data_x, data_y, s=len(data_x) * max_f, k=1)
    number_of_knots = slm.get_knots().size
    # determine max
    while slm.get_knots().size > desired_number_of_knots:
        max_f = max_f * 2
        slm = UnivariateSpline(data_x, data_y, s=len(data_x) * max_f, k=1)
    factor = max_f
    while not (
            abs(desired_number_of_knots - number_of_knots) <= threshold and number_of_knots <= desired_number_of_knots):
        slm = UnivariateSpline(data_x, data_y, s=len(data_x) * factor, k=1)
        if slm.get_knots().size > desired_number_of_knots:
            min_f = factor
            if max_f - factor < 0.01:
                max_f = max_f * 2
            factor = factor + (max_f - factor) / 2
        elif slm.get_knots().size < desired_number_of_knots:
            if closesed is None or desired_number_of_knots - closesed[
                1] > desired_number_of_knots - slm.get_knots().size:
                closesed = (slm, slm.get_knots().size)
            if factor - min_f < 0.01:
                min_f = min_f - 1
            max_f = factor
            factor = factor - (factor - min_f) / 2
        iteration += 1
        if iteration > max_iterations:
            print("cant find normal spline current number of knots:{} closeset:{}".format(number_of_knots,
                                                                                          -1 if closesed is None else
                                                                                          closesed[1]))
            if closesed is None or closesed[1] > desired_number_of_knots:
                raise RuntimeError("cant find normal spline")
            else:
                slm = closesed[0]
                break
    xHat = np.linspace(min(data_x), max(data_x), num=10000)
    yHat = slm(xHat)
    # plot the results
    plt.figure()
    plt.plot(data_x, data_y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.show()
    return slm

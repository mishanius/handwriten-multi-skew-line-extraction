import unittest
import cv2
from skimage.filters import threshold_niblack, apply_hysteresis_threshold
from skimage.filters.thresholding import _mean_std
from skimage.measure import regionprops
import math
from extractors.LineExtractorBase import LineExtractorBase
import numpy as np
import matplotlib.pyplot as plt

from extractors.MultiSkewExtractor import MultiSkewExtractor
from utils.approximate_using_piecewise_linear_pca import approximate_using_piecewise_linear_pca
from utils.label_broken_lines import concatenate_pixel_lists, label_broken_lines


class MultiSkewExtractorTests(unittest.TestCase):
    def test_split_lines(self):
        lines = np.array([[0, 1, 1, 1],
                          [1, 1, 0, 0],
                          [1, 0, 1, 1],
                          [0, 1, 1, 1],
                          [1, 1, 1, 1],
                          [0, 0, 1, 0]])
        plt.imshow(lines)
        plt.show()
        # print(MultiSkewExtractor.split_lines(lines, 0.3))

    def test_label_broken_lines(self):
        lines2split = np.array([[0, 0, 0, 1],
                                [0, 0, 1, 0],
                                [1, 1, 0, 0],
                                [1, 0, 2, 0],
                                [0, 0, 2, 0],
                                [0, 0, 2, 0]])

        broken_lines = np.array([[0, 0, 0, 1],
                                 [0, 0, 1, 0],
                                 [1, 0, 0, 0],
                                 [1, 0, 2, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 2, 0]])
        res = label_broken_lines(lines2split, broken_lines, 2)
        assert np.array_equal(res[0], np.array([[0., 0., 0., 3.],
                                                [0., 0., 3., 0.],
                                                [3., 3., 0., 0.],
                                                [3., 0., 4., 0.],
                                                [0., 0., 4., 0.],
                                                [0., 0., 4., 0.]]))

    def test_concatenate_pixel_lists(self):
        image_mock = np.array([[0, 0, 0, 1],
                               [0, 0, 1, 0],
                               [1, 1, 0, 0],
                               [1, 0, 2, 0],
                               [0, 0, 2, 0],
                               [0, 0, 2, 0]])
        props = regionprops(image_mock)
        res = concatenate_pixel_lists(props)
        assert np.array_equal(res, np.array([[0., 3., 0.],
                                             [1., 2., 0.],
                                             [2., 0., 0.],
                                             [2., 1., 0.],
                                             [3., 0., 0.],
                                             [3., 2., 1.],
                                             [4., 2., 1.],
                                             [5., 2., 1.]]))

    def test_approximate_using_piecewise_linear_pca(self):
        image_mock = np.array([[0, 0, 0, 1],
                               [0, 0, 1, 0],
                               [1, 1, 0, 0],
                               [1, 0, 2, 0],
                               [0, 0, 2, 0],
                               [0, 0, 2, 0]])

        fitting = approximate_using_piecewise_linear_pca(image_mock, 2, [], None)
        print(fitting)

    def test_morphology_tests(self):
        im = cv2.imread("../circles.png", 0)
        # im[im > 1] = [1]
        # skeleton = skeletonize(im)
        plt.imshow(im)
        plt.show()

    def test_apply_filters_functionality(self):
        cm = plt.get_cmap('gray')
        kw = {'cmap': cm, 'interpolation': 'none', 'origin': 'upper'}
        im = cv2.imread('ms_25_short.png', 0)
        angles = np.arange(0, 155, 25)
        scales = [16.8, 22.5]
        orient, _, response = MultiSkewExtractor.filter_document(im, scales, angles)
        print(response[99:119, 99:119])
        print("\norient:{}\n\n".format(orient[99:119, 99:119]))
        response = np.double(response)
        m, s = _mean_std(response, int(math.ceil(22.5) * 2 + 1))
        print("\nmean:{}\n\n".format(m[99:119, 99:119]))
        print("\nstd:{}\n\n".format(s[99:119, 99:119]))
        high = 22
        low = 8
        thresh_niblack2 = np.divide((response - m), s) * 20

        # plt.subplot(1, 3, 3)
        # plt.imshow(thresh_niblack2, **kw)
        # plt.title('thresh_niblack2')
        lines = apply_hysteresis_threshold(thresh_niblack2, low, high)
        print("\nlines:{}\n\n".format(lines[99:119, 99:119]))
        # plt.subplot(1, 3, 1)
        # plt.imshow(response, **kw)
        # plt.title('response')

        plt.subplot(1, 1, 1)
        plt.imshow(lines, **kw)
        plt.title('lines')
        plt.show()
        print("done")

    @staticmethod
    def dummy_func(a, b, c, theta, e, f):
        res = np.full((4, 4), 0)
        if theta == 1:
            res[0, :] = [17]
        if theta == 2:
            res[0, :] = [10]
            res[2, :] = [10]
        if theta == 3:
            res[1, :] = [15]
            res[3, :] = [16]
        return res

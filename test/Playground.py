import unittest
import cv2
from numpy.ma import arange
from skimage.filters import threshold_niblack, apply_hysteresis_threshold
from skimage.filters.thresholding import _mean_std
from numpy import linspace, exp
from numpy.random import randn
from scipy.interpolate import UnivariateSpline
from skimage.measure import regionprops
from skimage.morphology import skeletonize

from anigauss.matlabicani import anigauss
from plantcv import plantcv as pcv
from extractors.LineExtractorBase import LineExtractorBase
# from extractors.MultiSkewExtractor import MultiSkewExtractor
import numpy as np
import matplotlib.pyplot as plt

from extractors.MultiSkewExtractor import MultiSkewExtractor
from permuteLabels import permuteLabels
from utils.approximate_using_piecewise_linear_pca import approximate_using_piecewise_linear_pca
from utils.debugble_decorator import CacheSwitch
from utils.label_broken_lines import concatenate_pixel_lists, label_broken_lines
from utils.local_orientation_label_cost import local_orientation_label_cost


class Playground(unittest.TestCase):

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
                                                [4., 4., 0., 0.],
                                                [4., 0., 5., 0.],
                                                [0., 0., 5., 0.],
                                                [0., 0., 6., 0.]]))

    def test_concatenate_pixel_lists(self):
        image_mock = np.array([[0, 0, 0, 1],
                               [0, 0, 1, 0],
                               [1, 1, 0, 0],
                               [1, 0, 2, 0],
                               [0, 0, 2, 0],
                               [0, 0, 2, 0]])
        props = regionprops(image_mock)
        res = concatenate_pixel_lists(props)
        assert np.array_equal(res, np.array([[0., 3., 1.],
                                             [1., 2., 1.],
                                             [2., 0., 1.],
                                             [2., 1., 1.],
                                             [3., 0., 1.],
                                             [3., 2., 2.],
                                             [4., 2., 2.],
                                             [5., 2., 2.]]))

    def test_whole_flow(self):
        angles = np.arange(0, 155, 25)
        multi_extractor = MultiSkewExtractor('test/ms_25_short.png')
        multi_extractor.extract_lines(angles)
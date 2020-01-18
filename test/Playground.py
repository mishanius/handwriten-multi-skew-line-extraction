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

from anigauss.ani import circletest
from anigauss.ani import anigauss
from plantcv import plantcv as pcv
from extractors.LineExtractorBase import LineExtractorBase
# from extractors.MultiSkewExtractor import MultiSkewExtractor
import numpy as np
import matplotlib.pyplot as plt

from extractors.MultiSkewExtractor import MultiSkewExtractor
from utils.approximate_using_piecewise_linear_pca import approximate_using_piecewise_linear_pca
from utils.label_broken_lines import concatenate_pixel_lists, label_broken_lines


class Playground(unittest.TestCase):
    # def test_basic_functionality(self):
    #     delta_theta = 25
    #     theta = arange(0, 180, delta_theta)
    #     multi: MultiSkewExtractor = MultiSkewExtractor("ms_25_short.png")
    #     multi.extract_lines(theta)
    #     print("ok")
    def test_splines2(self):
        from scipy.interpolate import interp1d
        from scipy.optimize import fmin
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0., 0.308, 0.55, 0.546, 0.44])
        f = interp1d(x, y, kind='linear', bounds_error=False)

    def test_split_lines(self):
        lines = np.array([[0, 1, 1, 1],
                          [1, 1, 0, 0],
                          [1, 0, 1, 1],
                          [0, 1, 1, 1],
                          [1, 1, 1, 1],
                          [0, 0, 1, 0]])
        print(MultiSkewExtractor.split_lines(lines, 0.3))



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
        # image_mock = np.array([[10, 11, 12, 13],
        #                        [14, 15, 16, 17],
        #                        [18, 19, 20, 21],
        #                        [22, 23, 24, 25]])
        # [res, response] = LineExtractorBase.apply_filters(image_mock, (4, 4), 10, theta=[1, 2, 3, 4, 5],
        #                                                   func_to_apply=Playground.dummy_func)
        # assert np.array_equal(response, np.array([[17., 17., 17., 17.],
        #                                           [15., 15., 15., 15.],
        #                                           [10., 10., 10., 10.],
        #                                           [16., 16., 16., 16.]]))
        # assert np.array_equal(res, np.array([[0., 0., 0., 0.],
        #                                      [2., 2., 2., 2.],
        #                                      [1., 1., 1., 1.],
        #                                      [2., 2., 2., 2.]]))
        # #     using the actual anigause function
        # image_mock = np.array([[100, 200, 300, 400, 500],
        #                        [100, 200, 300, 400, 500],
        #                        [100, 200, 300, 400, 500],
        #                        [100, 200, 300, 400, 500],
        #                        [100, 200, 300, 400, 500]], dtype=np.int32)
        im = cv2.imread('ms_25_short.png', 0)
        nangles = 32
        angles = np.arange(0, 155, 25)
        [res, onlyfilter_response] = LineExtractorBase.apply_filters(im, (len(im), len(im[1])), 21.8632, theta=angles)
        print(res[100:120,100:120])
        print("{}-{}".format(angles[5], angles[4]))
        plt.subplot(1, 1, 1)
        plt.imshow(onlyfilter_response, **kw)
        plt.title('onlyfilter_response')
        plt.show()
        # _, _, response = MultiSkewExtractor.filter_document(im, [12.2, 16.8], angles)
        # response = np.double(response)
        # m, s = _mean_std(response, int(16.8) * 2 + 1)
        # high = 22
        # low = 8
        # thresh_niblack2 = np.divide((response - m), s) * 20
        # thresh_niblack = threshold_niblack(response, 16 * 2 + 1, 0.2)
        # # binary_niblack = response > thresh_niblack
        # lines = apply_hysteresis_threshold(thresh_niblack2, low, high)
        # scale = 16.8
        # eta = 3
        # plt.imshow(res, **kw)
        # plt.title('orientation')
        # a = anigauss(im, scale, eta * scale, 12, 2, 0)
        # plt.subplot(2, 3, 1)
        # plt.imshow(thresh_niblack2,**kw)
        # plt.title('niblack')

        # plt.subplot(2, 3, 2)
        # plt.imshow(response, **kw)
        # plt.title('response')

        # plt.subplot(2, 3, 3)
        # plt.imshow(lines, **kw)
        # plt.title('lines')
        #
        # plt.subplot(2, 3, 4)
        # plt.imshow(lines, **kw)
        # plt.title('lines')
        #
        # plt.subplot(2, 3, 5)
        # plt.imshow(thresh_niblack2, **kw)
        # plt.title('thresh_niblack2')
        # plt.show()

        # circletest()
        print("done")

    def test_apply_filter_test(self):
        cm = plt.get_cmap('gray')
        kw = {'cmap': cm, 'interpolation': 'none', 'origin': 'upper'}
        im = np.full((10,10),0)
        im[:,0:10] = [255.]
        # im[1,:] = [1]
        angles = np.arange(0, 155, 25)
        r = anigauss(im, 21.8632, 3 * 21.8632, angles[0], 2, 0)
        [res, onlyfilter_response] = LineExtractorBase.apply_filters(im, (len(im), len(im[1])), 21.8632, theta=angles)
        # _, _, response = LineExtractorBase.filter_document(im, [16.8,22.5], angles)
        # response[0:10,0:10]=[0]
        print(r)
        # print("{}-{}".format(angles[0], angles[3]))
        # response[response>255]=[255]
        # response[response < 0] = [0]
        # plt.subplot(1, 2, 1)
        # plt.imshow(response,**kw)
        # plt.title('response')
        # plt.subplot(1, 2, 2)
        # plt.imshow(im,**kw)
        # plt.title('original')
        # plt.show()

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

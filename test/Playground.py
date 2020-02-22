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
from utils.label_broken_lines import concatenate_pixel_lists, label_broken_lines
from utils.local_orientation_label_cost import local_orientation_label_cost


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

    def test_whole_flow(self):
        angles = np.arange(0, 155, 25)
        multi_extractor = MultiSkewExtractor('test/ms_25_short.png')
        multi_extractor.extract_lines(angles)

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
        # [res, onlyfilter_response] = LineExtractorBase.apply_filters(im, (len(im), len(im[1])), 21.8632, theta=angles)
        # print(res[100:120,100:120])
        # print("{}-{}".format(angles[5], angles[4]))
        # plt.subplot(1, 1, 1)
        # plt.imshow(onlyfilter_response, **kw)
        # plt.title('onlyfilter_response')
        # plt.show()
        _, _, response = MultiSkewExtractor.filter_document(im, [12.2, 16.8], angles)
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
        im = np.full((500, 500), 0)
        im[:, 0:20] = [255.]
        # im[1,:] = [1]
        angles = np.arange(0, 155, 25)
        # r = anigauss(im, 21.8632, 3 * 21.8632, 0, 2, 0)
        scale = 21.8632
        # [res, onlyfilter_response] = LineExtractorBase.apply_filters(im, (len(im), len(im[1])), 21.8632, theta=angles)
        res, _, onlyfilter_response = LineExtractorBase.filter_document(im, [16.8, 22.5], angles)
        # print(onlyfilter_response[99:109, 99:109])
        # onlyfilter_response = (scale * scale * 3) ** (2 / 2) * onlyfilter_response
        # onlyfilter_response = onlyfilter_response.reshape((500,500))
        print(res[99:119, 99:119])
        print(onlyfilter_response[99:109, 99:109])

        # response[0:10,0:10]=[0]
        # print("{}-{}".format(angles[0], angles[3]))
        # response[response>255]=[255]
        # response[response < 0] = [0]
        plt.subplot(1, 2, 1)
        plt.imshow(onlyfilter_response, **kw)
        plt.title('onlyfilter_response')
        plt.subplot(1, 2, 2)
        plt.imshow(im, **kw)
        plt.title('original')
        plt.show()


    def test_compute_label_cost(self):
        img = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 0, 0, 0, 1, 1, 0, 1],
             [1, 1, 0, 0, 0, 1, 1, 0, 1],
             [1, 1, 0, 0, 0, 1, 1, 0, 1],
             [0, 1, 1, 0, 0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 0, 0],
             [1, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 0, 0, 0]])

        labled_lines_original = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 3],
             [1, 1, 0, 0, 0, 2, 2, 0, 3],
             [1, 1, 0, 0, 0, 2, 2, 0, 3],
             [1, 1, 0, 0, 0, 2, 2, 0, 3],
             [0, 1, 1, 0, 0, 2, 2, 0, 0],
             [0, 0, 1, 1, 0, 2, 2, 0, 0],
             [0, 0, 0, 0, 0, 2, 2, 0, 0],
             [2, 2, 2, 2, 2, 2, 0, 0, 0],
             [0, 2, 2, 2, 0, 0, 0, 0, 0]])

        labeled_lines = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 2],
             [1, 1, 0, 0, 0, 4, 4, 0, 2],
             [1, 1, 0, 0, 0, 4, 4, 0, 2],
             [1, 1, 0, 0, 0, 4, 4, 0, 2],
             [0, 1, 1, 0, 0, 4, 4, 0, 0],
             [0, 0, 1, 1, 0, 4, 4, 0, 0],
             [0, 0, 0, 0, 0, 4, 4, 0, 0],
             [3, 3, 3, 3, 3, 3, 0, 0, 0],
             [0, 3, 3, 3, 0, 0, 0, 0, 0]])

        char_range = [2, 3, 4]
        theta = [0, 25, 50, 75, 90, 120, 140, 160, 180]
        max_orientation, _, max_response = MultiSkewExtractor.filter_document(img, char_range,
                                                                              theta)
        print("max_orientation:\n{},\n max_response:\n{}\n".format(max_orientation, max_response))
        labeled_lines_num = 4
        original_labeling = 3
        intact_lines_num = 2
        _, old_lines = MultiSkewExtractor.niblack_pre_process(max_response, 2 * np.round(char_range[1]) + 1, img)
        labeled_lines, lebeled_lines_num, intact_lines_num = MultiSkewExtractor.split_lines(old_lines, char_range[1])
        cost = MultiSkewExtractor.compute_line_label_cost(labled_lines_original, labeled_lines, labeled_lines_num,
                                                          intact_lines_num,
                                                          max_orientation, max_response, theta, radius_constant=2)
        assert np.array_equal(np.round(cost, decimals=3), np.array([[1.271], [1.822], [10.795], [1203040.833], [0.]]))
        _, _, new_lines = MultiSkewExtractor.post_process_by_mfr(labled_lines_original, original_labeling,
                                                                 labeled_lines,
                                                                 labeled_lines_num, cost, char_range)
        permuteLabels(new_lines)

        print("res:{}".format(cost))

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

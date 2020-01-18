from skimage.color import label2rgb
from skimage.filters import apply_hysteresis_threshold
from skimage.filters.thresholding import _mean_std
from skimage.measure import regionprops
from skimage.morphology import reconstruction
import cv2
from plantcv import plantcv as pcv
from skimage.morphology import thin, skeletonize
from extractors.LineExtractorBase import LineExtractorBase
from scipy.ndimage import label as bwlabel
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import generic_filter, uniform_filter
from skimage.filters import (threshold_niblack)

from utils.approximate_using_piecewise_linear_pca import approximate_using_piecewise_linear_pca
from utils.label_broken_lines import label_broken_lines


class MultiSkewExtractor(LineExtractorBase):

    def extract_lines(self, theta=0):
        max_orientation, _, max_response = MultiSkewExtractor.filter_document(self.image_to_process, self.char_range,
                                                                              theta)
        self.__niblack_pre_process(max_response, 33)

    def __niblack_pre_process(self, max_response, n):
        im = np.double(max_response)
        # local_sum = convolve2d(np.ones((n,n)), im)
        # local_num = convolve2d(np.ones(n,n), im * 0 + 1)
        # local_mean = local_sum/ local_num
        # local_mean[np.isinf(local_mean)] = 0
        # local_std = self.__window_stdev(im, n)
        #
        # pp_process = ((im - local_mean) / local_std) * 20
        #
        # high = 22
        # low = 8
        # lines = apply_hysteresis_threshold(pp_process, low, high)
        m, s = _mean_std(im, int(16.8) * 2 + 1)
        high = 22
        low = 8
        thresh_niblack2 = np.divide((im - m), s) * 20
        lines = apply_hysteresis_threshold(thresh_niblack2, low, high)
        lines = reconstruction(np.logical_and(self.bin_image, lines), lines, method='dilation')

        labled_lines, _ = bwlabel(lines)
        r = label2rgb(labled_lines, bg_color=(0, 0, 0))
        # cv2.imwrite('multi_skew_image.png', r)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', r)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # def __window_stdev(self, X, window_size):
    #     c1 = uniform_filter(X, window_size, mode='reflect')
    #     c2 = uniform_filter(X * X, window_size, mode='reflect')
    #     return np.sqrt(c2 - c1 * c1)
    @staticmethod
    def split_lines(lines, max_scale):
        labeled_lines, num_of_labeles = bwlabel(lines)
        fitting = approximate_using_piecewise_linear_pca(labeled_lines, num_of_labeles, [], 0)
        indices = (fitting < 0.8 * max_scale) | (fitting == np.inf)
        line_indices = np.argwhere(indices)
        non_line_indices = np.argwhere(np.logical_not(indices))
        non_line_indices = non_line_indices+[1]
        intact_lines, intact_lines_num = bwlabel(np.isin(labeled_lines, line_indices))
        lines2split = np.isin(labeled_lines, non_line_indices)
        labeled_lines_num = intact_lines_num
        skel = pcv.morphology.skeletonize(lines2split)
        branch_pts = pcv.morphology.find_branch_pts(skel_img=skel)
        # TODO check correct kernel
        branch_pts_fat = pcv.dilate(branch_pts, ksize=5, i=1)
        broken_lines = np.logical_and(skel, np.logical_not(branch_pts_fat))
        temp, labeled_lines_num = label_broken_lines(lines2split, broken_lines, labeled_lines_num)
        intact_lines[temp > 0] = temp[temp > 0]
        labeled_lines = intact_lines
        for i in range(labeled_lines_num):
            lbls, num = bwlabel(labeled_lines == i)
            if num > 1:
                props = regionprops(lbls)
                areas = [p.area for p in props]
                loc = np.argmax(areas)
                lbls[lbls == loc + 1] = 0
                labeled_lines[lbls > 0] = 0
        return labeled_lines, labeled_lines_num, labeled_lines_num

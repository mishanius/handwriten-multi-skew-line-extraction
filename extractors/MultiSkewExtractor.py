from plantcv.plantcv import dilate
from plantcv.plantcv.morphology import skeletonize, find_branch_pts
from skimage.color import label2rgb
from skimage.filters import apply_hysteresis_threshold
from skimage.filters.thresholding import _mean_std
from skimage.measure import regionprops
from skimage.morphology import reconstruction
from extractors.LineExtractorBase import LineExtractorBase
from scipy.ndimage import label as bwlabel
import numpy as np

from utils.approximate_using_piecewise_linear_pca import approximate_using_piecewise_linear_pca
from utils.label_broken_lines import label_broken_lines


class MultiSkewExtractor(LineExtractorBase):

    def extract_lines(self, theta=0):
        max_orientation, _, max_response = MultiSkewExtractor.filter_document(self.image_to_process, self.char_range,
                                                                              theta)
        _, old_lines = self.__niblack_pre_process(max_response, 2 * np.round(self.char_range[1]) + 1)

        labeled_lines, lebeled_lines_num, intact_lines_num = self.split_lines(old_lines, self.char_range[1])

    def __niblack_pre_process(self, max_response, n):
        im = np.double(max_response)
        m, s = _mean_std(im, int(16.8) * 2 + 1)
        high = 22
        low = 8
        thresh_niblack2 = np.divide((im - m), s) * 20
        lines = apply_hysteresis_threshold(thresh_niblack2, low, high)
        lines = reconstruction(np.logical_and(self.bin_image, lines), lines, method='dilation')
        labled_lines, _ = bwlabel(lines)
        r = label2rgb(labled_lines, bg_color=(0, 0, 0))
        return [thresh_niblack2, lines]

    @staticmethod
    def split_lines(lines, max_scale):
        labeled_lines, num_of_labeles = bwlabel(lines)
        fitting = approximate_using_piecewise_linear_pca(labeled_lines, num_of_labeles, [], 0)
        indices = (fitting < 0.8 * max_scale) | (fitting == np.inf)
        line_indices = np.argwhere(indices)
        non_line_indices = np.argwhere(np.logical_not(indices))
        non_line_indices = non_line_indices + [1]
        intact_lines, intact_lines_num = bwlabel(np.isin(labeled_lines, line_indices))
        lines2split = np.isin(labeled_lines, non_line_indices)
        labeled_lines_num = intact_lines_num
        skel = skeletonize(lines2split)
        branch_pts = find_branch_pts(skel_img=skel)
        # TODO check correct kernel
        branch_pts_fat = dilate(branch_pts, ksize=5, i=1)
        broken_lines = np.logical_and(skel, np.logical_not(branch_pts_fat))
        temp, labeled_lines_num = label_broken_lines(lines2split, broken_lines, labeled_lines_num)
        intact_lines[temp > 0] = temp[temp > 0]
        labeled_lines = intact_lines
        for i in range(1, labeled_lines_num + 1):
            lbls, num = bwlabel(labeled_lines == i)
            if num > 1:
                props = regionprops(lbls)
                areas = [p.area for p in props]
                loc = np.argmax(areas)
                lbls[lbls == loc + 1] = 0
                labeled_lines[lbls > 0] = 0
        return labeled_lines, labeled_lines_num, labeled_lines_num

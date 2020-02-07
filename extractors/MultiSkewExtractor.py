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
from utils.local_orientation_label_cost import local_orientation_label_cost


class MultiSkewExtractor(LineExtractorBase):

    def extract_lines(self, theta=0):
        max_orientation, _, max_response = MultiSkewExtractor.filter_document(self.image_to_process, self.char_range,
                                                                              theta)
        print("max_response:{}".format(max_response[99:120,99:120]))
        labled_lines_original, _ = bwlabel(self.bin_image)
        _, old_lines = self.__niblack_pre_process(max_response, 2 * np.round(self.char_range[1]) + 1)
        labeled_lines, lebeled_lines_num, intact_lines_num = self.split_lines(old_lines, self.char_range[1])
        print("labeled_lines:{}\n".format(labeled_lines[150:200,150:200]))
        print("lebeled_lines_num:{}\n".format(lebeled_lines_num))
        self.compute_line_label_cost(labled_lines_original, labeled_lines, lebeled_lines_num, intact_lines_num,
                                     max_orientation, max_response, theta)

    def __niblack_pre_process(self, max_response, n):
        im = np.double(max_response)
        # int(16.8) * 2 + 1
        m, s = _mean_std(im, 47)
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
        line_indices = np.argwhere(indices)[:,0] + [1]
        non_line_indices = np.argwhere(np.logical_not(indices))[:,0] + [1]
        intact_lines, intact_lines_num = bwlabel(np.isin(labeled_lines, line_indices))
        lines2split = np.isin(labeled_lines, non_line_indices)
        labeled_lines_num = intact_lines_num
        skel = skeletonize(lines2split)
        branch_pts = find_branch_pts(skel_img=skel)
        # TODO check correct kernel and iterations
        branch_pts_fat = dilate(branch_pts, ksize=5, i=1)
        broken_lines = np.logical_and(skel, np.logical_not(branch_pts_fat))
        lines2split = 1*lines2split
        temp, labeled_lines_num = label_broken_lines(1*lines2split, 1*broken_lines, labeled_lines_num)
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

    @staticmethod
    def compute_line_label_cost(raw_labeled_lines, labeled_lines, labeled_lines_num, intact_lines_num, max_orientation,
                                max_response, theta):
        acc = np.zeros((labeled_lines_num + 1, 1))
        mask_ = raw_labeled_lines.flatten()
        raw_labeled_lines_temp = labeled_lines.flatten()
        for index, (label, masked_label) in enumerate(zip(raw_labeled_lines_temp, mask_)):
            if label and masked_label:
                acc[label - 1] = acc[label - 1] + 1
        density_label_cost = np.exp(0.2 * np.amax(acc) / acc)
        density_label_cost[intact_lines_num + 1:] = [0]
        if intact_lines_num != labeled_lines_num:
            orientation_label_cost = local_orientation_label_cost(labeled_lines, labeled_lines_num, intact_lines_num,
                                                                  max_orientation, max_response,
                                                                  theta)
        else:
            orientation_label_cost = np.zeros(labeled_lines_num + 1, 1)
        return orientation_label_cost + density_label_cost

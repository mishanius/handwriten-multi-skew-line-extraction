import os

from plantcv.plantcv import dilate
from plantcv.plantcv.morphology import skeletonize, find_branch_pts
from skimage.color import label2rgb
from skimage.filters import apply_hysteresis_threshold
from skimage.filters.thresholding import _mean_std
from skimage.measure import regionprops
from skimage.morphology import reconstruction

from permuteLabels import permuteLabels
from RefineBinnaryOverlappingComponents import RefineBinnaryOverlappingComponents
from LineExtraction_GC_MRFminimization import line_extraction_GC
from computeLinesDC import compute_lines_data_cost
from computeNsSystem import computeNsSystem
from extractors.LineExtractorBase import LineExtractorBase
from scipy.ndimage import label as bwlabel
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.MetricLogger import MetricLogger
from utils.approximate_using_piecewise_linear_pca import approximate_using_piecewise_linear_pca
from utils.debug_utils import load_from_matlab
from utils.debugble_decorator import timed, partial_image, numpy_cached
from utils.draw_labels import draw_labels, labels2rgb
from utils.join_segments_skew import join_segments_skew
from utils.label_broken_lines import label_broken_lines
from utils.local_orientation_label_cost import local_orientation_label_cost
import scipy.io as sio

MATLAB_ROOT = "C:/Users/Itay/OneDrive - post.bgu.ac.il/academic/imageProcessing/LineExtraction2"
FULL_STRUCT = np.ones((3, 3))


class MultiSkewExtractor(LineExtractorBase):
    @timed(lgnm="extract_lines", log_max_runtime=True, verbose=True)
    def extract_lines(self, theta=0):
        cm = plt.get_cmap('gray')
        kw = {'cmap': cm, 'interpolation': 'none', 'origin': 'upper'}
        # max_orientation, _, max_response = MultiSkewExtractor.filter_document(self.image_to_process, self.char_range,
        #                                                                       theta)
        # labled_lines_original, num = bwlabel(self.bin_image, FULL_STRUCT)
        # _, old_lines = self.__niblack_pre_process(max_response, 2 * np.round(self.char_range[1]) + 1)
        # labeled_lines, lebeled_lines_num, intact_lines_num = self.split_lines(old_lines, self.char_range[1])

        l = sio.loadmat("{}/{}".format(MATLAB_ROOT, "L.mat"))
        labled_lines_original = l['L']
        LabeledLines = sio.loadmat(
            "{}/{}".format(MATLAB_ROOT, "LabeledLines.mat"))
        labeled_lines = LabeledLines['LabeledLines']
        num = 650
        LabeledLinesNum = sio.loadmat(
            "{}/{}".format(MATLAB_ROOT, "LabeledLinesNum.mat"))
        lebeled_lines_num = LabeledLinesNum['LabeledLinesNum'][0][0]
        #
        intactLinesNum = sio.loadmat(
            "{}/{}".format(MATLAB_ROOT, "intactLinesNum.mat"))
        intact_lines_num = intactLinesNum['intactLinesNum'][0][0]
        #
        max_orientation = sio.loadmat(
            "{}/{}".format(MATLAB_ROOT, "max_orientation.mat"))
        max_orientation = max_orientation['max_orientation']

        max_response = sio.loadmat(
            "{}/{}".format(MATLAB_ROOT, "max_response.mat"))
        max_response = max_response['max_response']
        #
        theta = sio.loadmat(
            "{}/{}".format(MATLAB_ROOT, "options-theta.mat"))
        theta = theta['gg'][0]

        new_lines = sio.loadmat(
            "{}/{}".format(MATLAB_ROOT, "newLines.mat"))
        new_lines = new_lines['newLines']


        # cost = self.compute_line_label_cost(labled_lines_original, labeled_lines, lebeled_lines_num, intact_lines_num,
        #                                     max_orientation, max_response, theta)

        # cost_old = sio.loadmat(
        #     "{}/{}".format(MATLAB_ROOT, "LabelCost_old.mat"))
        # cost = cost_old['LabelCost']
        #
        # _, _, new_lines = self.post_process_by_mfr(labled_lines_original, num, labeled_lines, lebeled_lines_num, cost,
        #                                            self.char_range)
        # new_lines, newLinesNum = permuteLabels(new_lines)

        newLinesNum =71
        combined_lines, newSegments = join_segments_skew(labled_lines_original, new_lines, newLinesNum,
                                                         self.char_range[1])


        r = labels2rgb(combined_lines.astype(np.uint8))
        plt.imshow(r)
        plt.title('combined_lines1')
        plt.show()

        combined_lines2 = sio.loadmat(
            "{}/{}".format(MATLAB_ROOT, "combinedLines.mat"))
        combined_lines2 = combined_lines2['combinedLines']
        r = labels2rgb(combined_lines2.astype(np.uint8))
        plt.imshow(r)
        plt.title('combined_lines2')
        plt.show()

        # combined_lines = combined_lines.astype(np.int32)
        # combinedLinesNum = np.amax(np.amax(combined_lines))

        # combinedLinesNum = 39
        # cost = sio.loadmat(
        #     "{}/{}".format(MATLAB_ROOT, "LabelCost.mat"))

        # cost = self.compute_line_label_cost(labled_lines_original, combined_lines, combinedLinesNum, combinedLinesNum,
        #                                     max_orientation, max_response, theta)

        # cost = cost['LabelCost']
        # results, _, new_lines = self.post_process_by_mfr(labled_lines_original, num, combined_lines, combinedLinesNum, cost,
        #                                            self.char_range)

        # new_lines2 = sio.loadmat("{}/{}".format(MATLAB_ROOT, "finalLines.mat"))
        # new_lines2 = new_lines2['finalLines']
        # r = labels2rgb(new_lines.astype(np.uint8))
        # plt.imshow(r)
        # plt.title('new_lines')
        # plt.show()
        #
        # r = labels2rgb(new_lines2.astype(np.uint8))
        # plt.imshow(r)
        # plt.title('fskel2')
        # plt.show()

        logger = MetricLogger()
        logger.flush_all()
        logger.flush_timings()
        print("done")

    @timed(lgnm="__niblack_pre_process", log_max_runtime=True, verbose=True)
    @partial_image(0, "niblack_pre_process")
    @numpy_cached
    def __niblack_pre_process(self, max_response, n):
        im = np.double(max_response)
        m, s = _mean_std(im, 47)
        high = 22
        low = 8
        thresh_niblack2 = np.divide((im - m), s) * 20
        lines = apply_hysteresis_threshold(thresh_niblack2, low, high)
        lines = reconstruction(np.logical_and(self.bin_image, lines), lines, method='dilation')
        labled_lines, _ = bwlabel(lines, FULL_STRUCT)
        r = label2rgb(labled_lines, bg_color=(0, 0, 0))
        return thresh_niblack2, lines

    @staticmethod
    @timed
    @partial_image(0, "niblack_pre_process")
    def niblack_pre_process(max_response, n, bin):
        im = np.double(max_response)
        # int(16.8) * 2 + 1
        m, s = _mean_std(im, 47)
        high = 22
        low = 8
        thresh_niblack2 = np.divide((im - m), s) * 20
        lines = apply_hysteresis_threshold(thresh_niblack2, low, high)
        lines = reconstruction(np.logical_and(bin, lines), lines, method='dilation')
        return thresh_niblack2, lines

    @staticmethod
    @timed(lgnm="split_lines", log_max_runtime=True, verbose=True)
    @partial_image(0, "split_lines")
    @numpy_cached
    def split_lines(lines, max_scale):
        labeled_lines, num_of_labeles = bwlabel(lines, FULL_STRUCT)
        # labeled_lines = sio.loadmat("{}/{}".format(MATLAB_ROOT, "labeled_lines.mat"))
        # labeled_lines = labeled_lines['L']
        num_of_labeles = 60
        fitting = approximate_using_piecewise_linear_pca(labeled_lines, num_of_labeles, [], 0)
        # fitting = sio.loadmat("{}/{}".format(MATLAB_ROOT, "fitting.mat"))
        # fitting = fitting['fitting']
        indices = (fitting < 0.8 * max_scale) | (fitting == np.inf)
        line_indices = np.argwhere(indices)[:, 0] + [1]
        non_line_indices = np.argwhere(np.logical_not(indices))[:, 0] + [1]
        intact_lines, intact_lines_num = bwlabel(np.isin(labeled_lines, line_indices), FULL_STRUCT)
        lines2split = np.isin(labeled_lines, non_line_indices)
        labeled_lines_num = intact_lines_num
        # TODO skeletonize loses data
        skel = skeletonize(lines2split)
        # skel = load_from_matlab('skel')
        # cm = plt.get_cmap('gray')
        # kw = {'cmap': cm, 'interpolation': 'none', 'origin': 'upper'}
        # plt.imshow(skel, **kw)
        # plt.title('fskel')
        # plt.show()

        lines2split = np.isin(labeled_lines, non_line_indices)
        # TODO find_branch_pts loses data
        branch_pts = find_branch_pts(skel_img=skel)
        branch_pts_fat = dilate(branch_pts, ksize=3, i=1)
        # branch_pts_fat = load_from_matlab('B_fat')
        # cm = plt.get_cmap('gray')
        # kw = {'cmap': cm, 'interpolation': 'none', 'origin': 'upper'}
        #
        # plt.imshow(branch_pts_fat, **kw)
        # plt.title('branch_pts_fat')
        # plt.show()

        broken_lines = np.logical_and(skel, np.logical_not(branch_pts_fat))

        # plt.imshow(broken_lines, **kw)
        # plt.title('broken_lines')
        # plt.show()

        lines2split = 1 * lines2split
        temp, labeled_lines_num = label_broken_lines(1 * lines2split, 1 * broken_lines, labeled_lines_num)
        intact_lines[temp > 0] = temp[temp > 0]
        labeled_lines = intact_lines

        for i in range(1, int(labeled_lines_num) + 1):
            lbls, num = bwlabel(labeled_lines == i, FULL_STRUCT)
            if num > 1:
                props = regionprops(lbls)
                areas = [p.area for p in props]
                loc = np.argmax(areas)
                lbls[lbls == loc + 1] = 0
                labeled_lines[lbls > 0] = 0

        # r = label2rgb(labeled_lines, bg_color=(0, 0, 0))
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 600, 600)
        # cv2.imshow('image', r)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return labeled_lines, labeled_lines_num, intact_lines_num

    @staticmethod
    @timed(lgnm="compute_line_label_cost", log_max_runtime=True, verbose=True)
    @numpy_cached
    def compute_line_label_cost(raw_labeled_lines, labeled_lines, labeled_lines_num, intact_lines_num, max_orientation,
                                max_response, theta, radius_constant=18):
        # acc = np.zeros((labeled_lines_num + 1, 1))
        mask_ = raw_labeled_lines.flatten()
        labeled_lines_temp = labeled_lines.flatten()
        masked_labeled_lines_temp = labeled_lines_temp[mask_ > 0]
        masked_labeled_lines_temp = masked_labeled_lines_temp[masked_labeled_lines_temp > 0]
        acc = np.bincount(masked_labeled_lines_temp)
        acc = acc.reshape((acc.shape[0], 1))
        if len(acc) != labeled_lines_num + 1:
            acc = np.append(acc, np.zeros((labeled_lines_num + 1 - len(acc), 1)), 0)
        acc[:-1] = acc[1:]
        # for index, (label, masked_label) in enumerate(zip(labeled_lines_temp, mask_)):
        #     if label and masked_label:
        #         acc[label - 1] = acc[label - 1] + 1
        density_label_cost = np.exp(0.2 * np.amax(acc) / acc)
        density_label_cost[intact_lines_num:] = [0]
        if intact_lines_num != labeled_lines_num:
            orientation_label_cost = local_orientation_label_cost(labeled_lines, labeled_lines_num, intact_lines_num,
                                                                  max_orientation, max_response,
                                                                  theta, radius_constant)
        else:
            orientation_label_cost = np.zeros((labeled_lines_num + 1, 1))
        return orientation_label_cost[:, 0] + density_label_cost

    @staticmethod
    @partial_image(2, "post_process_by_mfr")
    @numpy_cached
    def post_process_by_mfr(labeled_raw_components, raw_components_num, labeled_lines, labeled_lines_num, cost,
                            char_range):
        cc_sparse_ns = computeNsSystem(labeled_raw_components, raw_components_num)
        data_cost = compute_lines_data_cost(labeled_lines, labeled_lines_num, labeled_raw_components,
                                            raw_components_num,
                                            char_range[1])
        labels = line_extraction_GC(raw_components_num, labeled_lines_num, data_cost, cc_sparse_ns, cost)

        labels[labels == labeled_lines_num + 1] = 0
        # residual_lines = np.isin(labeled_lines, labels)
        labeled_lines[np.logical_not( np.isin(labeled_lines, labels))] = 0
        result = draw_labels(labeled_raw_components, labels)
        refined_ccs = RefineBinnaryOverlappingComponents(labeled_raw_components, raw_components_num, labeled_lines,
                                                         labeled_lines_num)
        temp_mask = refined_ccs > 0
        result[temp_mask] = refined_ccs[temp_mask]
        return result, labels, labeled_lines

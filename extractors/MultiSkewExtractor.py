from plantcv.plantcv import dilate
from plantcv.plantcv.morphology import skeletonize, find_branch_pts
from skimage.color import label2rgb
from skimage.filters import apply_hysteresis_threshold
from skimage.filters.thresholding import _mean_std
from skimage.measure import regionprops
from skimage.morphology import reconstruction
from RefineBinnaryOverlappingComponents import RefineBinnaryOverlappingComponents
from LineExtraction_GC_MRFminimization import line_extraction_GC
from computeLinesDC import compute_lines_data_cost
from computeNsSystem import computeNsSystem
from extractors.LineExtractorBase import LineExtractorBase
from scipy.ndimage import label as bwlabel
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.approximate_using_piecewise_linear_pca import approximate_using_piecewise_linear_pca
from utils.draw_labels import draw_labels
from utils.label_broken_lines import label_broken_lines
from utils.local_orientation_label_cost import local_orientation_label_cost



class MultiSkewExtractor(LineExtractorBase):

    def extract_lines(self, theta=0):
        cm = plt.get_cmap('gray')
        kw = {'cmap': cm, 'interpolation': 'none', 'origin': 'upper'}

        max_orientation, _, max_response = MultiSkewExtractor.filter_document(self.image_to_process, self.char_range,
                                                                              theta)
        print("max_response:{}".format(max_response[99:120, 99:120]))
        labled_lines_original, num = bwlabel(self.bin_image)
        _, old_lines = self.__niblack_pre_process(max_response, 2 * np.round(self.char_range[1]) + 1)
        labeled_lines, lebeled_lines_num, intact_lines_num = self.split_lines(old_lines, self.char_range[1])
        print("labeled_lines:{}\n".format(labeled_lines[150:200, 150:200]))
        print("lebeled_lines_num:{}\n".format(lebeled_lines_num))
        cost = self.compute_line_label_cost(labled_lines_original, labeled_lines, lebeled_lines_num, intact_lines_num,
                                     max_orientation, max_response, theta)
        print("finished cost !!")
        _,_ ,new_lines = self.post_process_by_mfr(labled_lines_original,num, labeled_lines, lebeled_lines_num, cost, self.char_range)
        plt.imshow(new_lines, **kw)
        plt.title('original new lines!!!')
        plt.show()

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
    def niblack_pre_process(max_response, n, bin):
        im = np.double(max_response)
        # int(16.8) * 2 + 1
        m, s = _mean_std(im, 47)
        high = 22
        low = 8
        thresh_niblack2 = np.divide((im - m), s) * 20
        lines = apply_hysteresis_threshold(thresh_niblack2, low, high)
        lines = reconstruction(np.logical_and(bin, lines), lines, method='dilation')
        return [thresh_niblack2, lines]

    @staticmethod
    def split_lines(lines, max_scale):
        cm = plt.get_cmap('gray')
        kw = {'cmap': cm, 'interpolation': 'none', 'origin': 'upper'}

        labeled_lines, num_of_labeles = bwlabel(lines)
        fitting = approximate_using_piecewise_linear_pca(labeled_lines, num_of_labeles, [], 0)
        indices = (fitting < 0.8 * max_scale) | (fitting == np.inf)
        line_indices = np.argwhere(indices)[:, 0] + [1]
        non_line_indices = np.argwhere(np.logical_not(indices))[:, 0] + [1]
        intact_lines, intact_lines_num = bwlabel(np.isin(labeled_lines, line_indices))
        lines2split = np.isin(labeled_lines, non_line_indices)
        labeled_lines_num = intact_lines_num
        skel = skeletonize(lines2split)

        intact_lines, intact_lines_num = bwlabel(np.isin(labeled_lines, line_indices))
        lines2split = np.isin(labeled_lines, non_line_indices)

        plt.imshow(1 * lines2split, **kw)
        plt.show()

        plt.imshow(1 * lines2split, **kw)
        plt.show()

        branch_pts = find_branch_pts(skel_img=skel)
        # TODO check correct kernel and iterations
        branch_pts_fat = dilate(branch_pts, ksize=5, i=1)

        # TODO intact not the same because of the spline fitting
        plt.imshow(intact_lines, **kw)
        plt.show()

        broken_lines = np.logical_and(skel, np.logical_not(branch_pts_fat))

        plt.imshow(branch_pts_fat, **kw)
        plt.show()

        lines2split = 1 * lines2split
        temp, labeled_lines_num = label_broken_lines(1 * lines2split, 1 * broken_lines, labeled_lines_num)
        intact_lines[temp > 0] = temp[temp > 0]
        labeled_lines = intact_lines

        r = label2rgb(temp, bg_color=(0, 0, 0))
        cv2.imshow('image', r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.imshow(intact_lines, **kw)
        plt.show()

        connectivity_struct = np.array([[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]])

        for i in range(1, int(labeled_lines_num) + 1):
            lbls, num = bwlabel(labeled_lines == i, connectivity_struct)
            if num > 1:
                props = regionprops(lbls)
                areas = [p.area for p in props]
                loc = np.argmax(areas)
                print(areas)
                lbls[lbls == loc + 1] = 0
                labeled_lines[lbls > 0] = 0

        r = label2rgb(labeled_lines, bg_color=(0, 0, 0))
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600, 600)
        cv2.imshow('image', r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return labeled_lines, labeled_lines_num, intact_lines_num

    @staticmethod
    def compute_line_label_cost(raw_labeled_lines, labeled_lines, labeled_lines_num, intact_lines_num, max_orientation,
                                max_response, theta, radius_constant=18):
        acc = np.zeros((labeled_lines_num + 1, 1))
        mask_ = raw_labeled_lines.flatten()
        raw_labeled_lines_temp = labeled_lines.flatten()
        for index, (label, masked_label) in enumerate(zip(raw_labeled_lines_temp, mask_)):
            if label and masked_label:
                acc[label - 1] = acc[label - 1] + 1
        density_label_cost = np.exp(0.2 * np.amax(acc) / acc)
        density_label_cost[intact_lines_num:] = [0]
        if intact_lines_num != labeled_lines_num:
            orientation_label_cost = local_orientation_label_cost(labeled_lines, labeled_lines_num, intact_lines_num,
                                                                  max_orientation, max_response,
                                                                  theta, radius_constant)
        else:
            orientation_label_cost = np.zeros(labeled_lines_num + 1, 1)
        return orientation_label_cost + density_label_cost

    @staticmethod
    def post_process_by_mfr(labeled_raw_components, raw_components_num, labeled_lines, labeled_lines_num, cost, char_range):
        cc_sparse_ns = computeNsSystem(labeled_raw_components, raw_components_num)
        data_cost = compute_lines_data_cost(labeled_lines, labeled_lines_num, labeled_raw_components,
                                            raw_components_num,
                                            char_range[1])
        labels = line_extraction_GC(raw_components_num, labeled_lines_num, data_cost, cc_sparse_ns, cost)

        labels[labels == labeled_lines_num + 1] = 0
        residual_lines = np.isin(labeled_lines, labels)
        labeled_lines[np.logical_not(residual_lines)] = 0
        result = draw_labels(labeled_raw_components, labels)
        refined_ccs = RefineBinnaryOverlappingComponents(labeled_raw_components, raw_components_num, labeled_lines, labeled_lines_num)
        temp_mask = refined_ccs > 0
        result[temp_mask] = refined_ccs[temp_mask]

        return [result, labels, labeled_lines]
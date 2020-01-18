import unittest
from scipy.ndimage import label as bwlabel
import numpy as np
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors

import gco
from PostProcessByMRF import drawLabels
from computeLinesDC import compute_lines_data_cost
from computeNsSystem import computeNsSystem


class Sanity(unittest.TestCase):
    def setUp(self):
        self.binarized_line_image = np.array([[1, 0, 1, 1],
                                              [0, 1, 0, 1],
                                              [0, 0, 0, 0],
                                              [1, 1, 1, 1]])
        self.lines = np.array([[1, 1, 1, 1],
                               [1, 1, 0, 0],
                               [0, 0, 0, 1],
                               [1, 1, 1, 1]])


class SanityTestCase(Sanity):
    def test_compute_lines_data_cost(self):
        binarized_line_image = np.array([[1, 0, 1],
                                         [0, 1, 0],
                                         [0, 0, 0],
                                         [1, 1, 1]])
        lines = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [0, 0, 0],
                          [1, 1, 1]])
        labeled_raw_lines, number_of_raw_lines = bwlabel(binarized_line_image)
        labeled_lines, number_of_labled_lines = bwlabel(lines)
        costs = compute_lines_data_cost(labeled_lines, number_of_labled_lines, labeled_raw_lines, number_of_raw_lines,
                                        10)
        print(costs)
        # assert np.array_equal(labels,np.array([
        #     [0.0,3.0,50.0],
        #     [0.0,3.0,50.0],
        #     [1.0, 2.0, 50.0],
        #     [3.0, 0.0, 50.0]
        # ]))
        gc = gco.GCO()
        gc.create_general_graph(4, 3)
        # gc.set_data_cost(np.array([di for di in data_cost.transpose()]))
        gc.set_data_cost(np.array([di for di in costs]))
        gc.expansion()
        labels = gc.get_labels() + 1
        print(labels)
        res = drawLabels(labeled_raw_lines, labels)
        print(res)

    def test_lines_data_cost(self):
        labeled_raw_lines, number_of_raw_lines = bwlabel(self.binarized_line_image)

        labeled_raw_lines_centroids = [[prop.centroid[0], prop.centroid[1]] for prop in regionprops(labeled_raw_lines)]

        labeled_lines, number_of_labled_lines = bwlabel(self.lines)

        labeled_lines_region_props = regionprops(labeled_lines)

        data_cost = np.empty((0, number_of_raw_lines), np.float32)
        for line_num, labeld_line_prop in enumerate(labeled_lines_region_props, start=0):
            line_coords = labeld_line_prop.coords
            nbrs = NearestNeighbors(1).fit(line_coords)
            dist, _ = nbrs.kneighbors(labeled_raw_lines_centroids)
            data_cost = np.append(data_cost, dist.transpose(), axis=0)
            print("dist:\n{}".format(dist.transpose()))
        gc = gco.GCO()
        gc.create_general_graph(number_of_raw_lines, number_of_labled_lines)
        gc.set_data_cost(np.array([di for di in data_cost.transpose()]))
        gc.expansion()
        labels = gc.get_labels()
        print(labels)
        assert np.array_equal(labels, np.array([0, 0, 0, 1]))


class NsSystemTest(Sanity):
    def setUp(self):
        self.binarized_line_image = np.array([[1, 0, 1],
                                              [0, 0, 0],
                                              [1, 0, 1],
                                              [0, 0, 0]])

    def test_compute_ns_system(self):
        labeled_raw_lines, number_of_raw_lines = bwlabel(self.binarized_line_image)
        ns_system = computeNsSystem(labeled_raw_lines, number_of_raw_lines)
        assert np.array_equal(ns_system, np.array([
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0]
        ]))


class LabelCostTest(Sanity):
    def setUp(self):
        self.binarized_line_image = np.array([[1, 0, 1],
                                              [0, 0, 0],
                                              [1, 0, 1],
                                              [0, 0, 0]])

    def test_basic(self):
        labeled_raw_lines, number_of_raw_lines = bwlabel(self.binarized_line_image)
        gc = gco.GCO()
        gc.create_general_graph(number_of_raw_lines, 3)
        gc.set_data_cost(np.array([[1,1,1], [1,1,1], [1,1,1], [1,1,1]]))
        # gc.set_smooth_cost(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]))
        gc.set_label_cost(np.array([100, 100, 50]))

        gc.expansion()
        labels = gc.get_labels()
        print(labels)
        assert np.array_equal(labels, np.array([2, 2, 2, 2]))

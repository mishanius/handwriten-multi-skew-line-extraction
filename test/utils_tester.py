from unittest import TestCase
import numpy as np

from utils.local_orientation_label_cost import local_orientation_label_cost


class UtilsFunctionsTester(TestCase):

    def test_local_orientation_label_cost(self):
        labeled_lines_num = 2
        intact_lines_num = 0
        max_orientation = np.array([[1, 0]])
        max_response = np.array([[10, 99]])
        theta = [0, 45]
        labeled_lines = np.array([[0, 0, 0, 1, 1],
                                  [0, 0, 1, 1, 0],
                                  [0, 1, 1, 0, 0],
                                  [0, 0, 0, 2, 2],
                                  [0, 0, 0, 2, 2]])
        # labeled_lines, labeled_lines_num, intact_lines_num, max_orientation,
        # max_response, theta
        res = local_orientation_label_cost(labeled_lines, labeled_lines_num, intact_lines_num, max_orientation,
                                           max_response, theta, radius_constant=2)
        assert round(res[0][0]) == 1011687
        assert res[1][0] == 10.
        assert res[2][0] == 0

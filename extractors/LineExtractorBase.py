from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

from anigauss.matlabicani import anigauss
import cv2

from estimateBinaryHeight import estimateBinaryHeight
from utils.debugble_decorator import numpy_cached


class LineExtractorBase(ABC):

    def __init__(self, image_path):
        self.image_to_process = cv2.imread(image_path, 0)
        self.bin_image = cv2.bitwise_not(self.image_to_process)
        self.char_range = estimateBinaryHeight(self.bin_image, 0.03)#[16.8632, 22.5972] #
        super().__init__()

    @abstractmethod
    def extract_lines(self, theta=0):
        pass

    @staticmethod
    def apply_filters(in_image, sz, scale, theta=None, eta=3,
                      func_to_apply=lambda a, b, c, d, e, f: anigauss(a, b, c, d, e, f)):
        if theta is None:
            theta = []
        max_responses = np.full((2, sz[0] * sz[1]), -np.inf)
        max_loc = np.full((1, sz[0] * sz[1]),  -np.inf)
        # responses = np.empty((len(theta), sz[0]*sz[1]))
        for index, t in enumerate(theta):
            f = func_to_apply(in_image, scale, eta * scale, t, 2, 0)
            max_responses[1, :] = f.flatten()
            max_loc[0, np.argmax(max_responses, 0) > 0] = [index]
            max_responses[0, :] = np.amax(max_responses, 0)

        res = np.reshape(max_loc, (sz[0], sz[1]))
        response = np.reshape(max_responses[0, :], (sz[0], sz[1]))
        return [res, response]

    @staticmethod
    @numpy_cached
    def filter_document(im, scales, theta=0):
        # im = np.uint8(im) * 255
        sz = [len(im), len(im[0])]

        orientation_map = np.full((1, sz[0] * sz[1]), -np.inf)
        response_map = np.full((2, sz[0] * sz[1]), -np.inf)
        scales_res = np.full((1, sz[0] * sz[1]), -np.inf)

        eta = 3
        gamma = 2
        for scale in scales:
            [orientation, response] = LineExtractorBase.apply_filters(im, sz, scale, theta)

            flat_response = response.flatten()
            response_map[1,:] = (scale * scale * eta) ** (gamma / 2) * flat_response

            val = np.amax(response_map, 0)

            loc = np.argmax(response_map, 0).reshape((1,sz[0] * sz[1]))
            response_map[0,:] = val
            orientation = np.reshape(orientation,(1,sz[0] * sz[1]))
            orientation_map[loc == 1] = orientation[loc == 1]
            scales_res[loc == 1] = [scale]
            # for index in indexesOfLoc1:
            #     orientation_map[index] = orientation[index]
            #     scales_res[index] = scale

        mat_size = (sz[0], sz[1])
        scales_res = scales_res.reshape(mat_size)
        max_response = response_map[0].reshape(mat_size)
        print("max_response:{}".format(max_response))
        return np.reshape(orientation_map,mat_size), scales_res, max_response

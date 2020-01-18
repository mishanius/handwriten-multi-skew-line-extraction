from abc import ABC, abstractmethod
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from anigauss.ani import anigauss
import cv2

from estimateBinaryHeight import estimateBinaryHeight


class LineExtractorBase(ABC):

    def __init__(self, image_path):
        self.image_to_process = cv2.imread(image_path, 0)
        self.bin_image = cv2.bitwise_not(self.image_to_process)
        self.char_range = estimateBinaryHeight(self.bin_image)
        super().__init__()

    @abstractmethod
    def extract_lines(self, theta=0):
        pass

    @staticmethod
    def apply_filters(in_image, sz, scale, theta=None, eta=3, func_to_apply = lambda a, b, c, d, e, f:anigauss(a, b, c, d, e, f)):
        if theta is None:
            theta = []
        max_responses = np.full((2, sz[0] * sz[1]), -np.inf)
        max_loc = np.full((1, sz[0] * sz[1]), 0)
        # responses = np.empty((len(theta), sz[0]*sz[1]))
        for index, t in enumerate(theta):
            max_responses[1,:] = func_to_apply(in_image, scale, eta * scale, t, 2, 0).flatten()
            max_loc[0, np.argmax(max_responses, 0) > 0] = [index]
            max_responses[0, :] = np.amax(max_responses,0)

        res = np.reshape(max_loc, (sz[0], sz[1]))
        response = np.reshape(max_responses[0,:], (sz[0], sz[1]))

        return [res, response]

    @staticmethod
    def filter_document(im, scales, theta=0):
        # im = np.uint8(im) * 255
        sz = [len(im), len(im[0])]

        orientation_map = np.full((1, sz[0] * sz[1]), -np.inf)
        response_map = np.full((2, sz[0] * sz[1]), -np.inf)
        scales_res = np.full((1, sz[0] * sz[1]), -np.inf)

        eta = 3
        gamma = 2
        cm = plt.get_cmap('binary')
        for scale in scales:
            [orientation, response] = LineExtractorBase.apply_filters(im, sz, scale, theta)
            kw = {'cmap': cm, 'interpolation': 'none', 'origin': 'upper'}
            plt.subplot(2, 3, 5)
            plt.imshow(response, **kw)
            plt.title('response2')
            flat_response = np.matrix(response).flatten()
            response_map[1] = (scale * scale * eta) ** (gamma / 2) * flat_response

            val = np.amax(response_map, 0)

            loc = np.argmax(response_map, 1)

            response_map[0] = val
            indexesOfLoc1 = np.where(loc == 1)[0].tolist()

            for index in indexesOfLoc1:
                orientation_map[index] = orientation[index]
                scales_res[index] = scale

        mat_size = (sz[0], sz[1])
        scales_res = np.array(scales_res).reshape(mat_size)
        flat_response_map = np.matrix(response_map[0]).flatten()
        max_response = flat_response_map.reshape(mat_size)

        return [np.array(orientation_map).reshape(mat_size), scales_res, max_response]

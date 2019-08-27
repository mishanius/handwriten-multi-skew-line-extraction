import numpy as np
from applyFilters import applyFilters
def filterDocument(im,scales):
    im = np.uint8(im) * 255
    sz = [len(im),len(im[0])]



    orientation_map = np.full((1, sz[0]*sz[1]), -np.inf)
    response_map = np.full((2, sz[0]*sz[1]), -np.inf)
    scales_res = np.full((1, sz[0]*sz[1]), -np.inf)

    eta = 3
    gamma =2

    for scale in scales:
        [orientation, response] = applyFilters(im, sz, scale)
        flat_response = np.matrix(response).flatten()
        response_map[1] = (scale * scale * eta) ** (gamma/2)*flat_response

        val = np.amax(response_map, 0)

        loc = np.argmax(response_map, 1)

        response_map[0]= val
        indexesOfLoc1 = np.where(loc == 1)[0].tolist()

        for index in indexesOfLoc1:
            orientation_map[index] = orientation[index]
            scales_res[index] = scale


    mat_size = (sz[0],sz[1])
    max_orientation = np.array(orientation_map).reshape(mat_size)
    scales_res = np.array(scales_res).reshape(mat_size)
    flat_response_map = np.matrix(response_map[0]).flatten()
    max_response = flat_response_map.reshape(mat_size)


    return [max_orientation, scales_res, max_response]
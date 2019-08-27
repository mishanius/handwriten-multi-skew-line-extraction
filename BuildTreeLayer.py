import numpy as np

def BuildTreeLayer(parentL, parentNum, childL, childNum):
    flat_childL = np.matrix(childL).flatten().tolist()[0]
    flat_parentL = np.matrix(parentL).flatten().tolist()[0]

    res = np.zeros(childNum).tolist()

    for i in range(len(flat_childL)):
        if flat_childL[i] != 0:
            res[flat_childL[i] - 1] = flat_parentL[i]

    layer = np.zeros((1, parentNum))
    if len(res)==0:
        return layer
    temp = np.histogram(res, len(res))
    max_mul = np.max(temp[0])

    layer = np.zeros((max_mul,parentNum ))

    for i in range(childNum):
        indexInRes = res[i]
        idx = np.nonzero(column(layer, indexInRes - 1) == 0)[0][0]
        layer[idx, indexInRes - 1] = i

    return layer

def column(matrix, i):

    return np.array([row[i] for row in matrix])
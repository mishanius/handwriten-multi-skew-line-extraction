import numpy as np
import gco


def LineExtraction_GC_MRFminimization(numLines, num, CCsparseNs, Dc, LabelCost):
    K = numLines + 1

    edgeWeights = computeEdgeWeights(CCsparseNs)

    gc = gco.GCO()

    gc.create_general_graph(num, K, False)

    threshHold = 10 ** 7
    Dc[Dc > threshHold] = threshHold - 1
    LabelCost[LabelCost > threshHold] = threshHold - 1
    LabelCost = np.array(np.round(LabelCost), dtype=np.int32)

    sparse = []
    for rowIndex in range(num):
        for colIndex in range(num):
            if edgeWeights[rowIndex, colIndex] != 0 and 0 < rowIndex < colIndex < num:
                sparse.append(((rowIndex, colIndex), np.int32(edgeWeights[rowIndex, colIndex])))

    gc.set_data_cost(np.int32(np.round(Dc.transpose())))
    Smooth_cost = np.int32((np.ones(K) - np.eye(K)))
    gc.set_smooth_cost(Smooth_cost)
    gc.set_label_cost(LabelCost)

    for currEdge in sparse:
        gc.set_neighbor_pair(currEdge[0][0], currEdge[0][1], currEdge[1])

    gc.expansion()
    Labels = gc.get_labels()
    gc.destroy_graph()
    return Labels


def line_extraction_GC(num_connected_componants, num_of_lines, data_cost, cc_sparse_ns=None, label_cost=None):
    gc = gco.GCO()
    gc.create_general_graph(num_connected_componants, num_of_lines, False)
    gc.set_data_cost(np.array([di for di in data_cost]))
    sparse = []
    if cc_sparse_ns is not None:
        edgeWeights = computeEdgeWeights(cc_sparse_ns)
        for row_index in range(cc_sparse_ns.shape[0]):
            for col_index in range(cc_sparse_ns.shape[1]):
                if edgeWeights[row_index, col_index] != 0 and 0 < row_index < col_index < num_connected_componants:
                    sparse.append(((row_index, col_index), np.int32(edgeWeights[row_index, col_index])))

        smooth_cost = np.int32((np.ones(num_of_lines) - np.eye(num_of_lines)))
        gc.set_smooth_cost(smooth_cost)

    if label_cost is not None:
        threshHold = 10
        data_cost[data_cost > threshHold] = threshHold - 1
        label_cost[label_cost > threshHold] = threshHold - 1
        label_cost = np.array(np.round(label_cost), dtype=np.int32)

        gc.set_label_cost(label_cost)

    for currEdge in sparse:
        gc.set_neighbor_pair(currEdge[0][0], currEdge[0][1], currEdge[1])

    gc.expansion()
    result_labels = gc.get_labels() + 1
    gc.destroy_graph()
    return result_labels


def computeEdgeWeights(W):
    #   W = np.array(W).transpose()
    gcScale = 100
    beta = 1 / (0.5 * np.mean(W[W > 0]))
    edgeWeights = np.exp(-beta * W)
    edgeWeights[edgeWeights >= 1] = 0
    edgeWeights = np.round(gcScale * edgeWeights)
    return edgeWeights

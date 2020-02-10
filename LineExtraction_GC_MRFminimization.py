import numpy as np
import gco
import scipy.io
from subprocess import Popen, PIPE

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


def line_extraction_GC(num_connected_componants, num_of_lines, data_cost, cc_sparse_ns=None, label_cost=None, should_solve_via_csv=True):
    sparse = []
    if cc_sparse_ns is not None:
        edgeWeights = computeEdgeWeights(cc_sparse_ns)
        for col_index in range(cc_sparse_ns.shape[1]):
            for row_index in range(cc_sparse_ns.shape[0]):
                if edgeWeights[row_index, col_index] != 0 and 0 <= row_index < col_index < num_connected_componants:
                    sparse.append(((np.int32(row_index), np.int32(col_index)), np.int32(edgeWeights[row_index, col_index])))

    smooth_cost = np.int32((np.ones(num_of_lines+1) - np.eye(num_of_lines+1)))

    threshHold = 10000000
    data_cost[data_cost > threshHold] = threshHold - 1
    data_cost = np.int32(np.around(data_cost))
    if label_cost is not None:
        label_cost[label_cost > threshHold] = threshHold - 1
        label_cost = np.array(np.around(label_cost), dtype=np.int32)

    neighbors = np.empty((len(sparse), 3))

    #for debug in matlab
    scipy.io.savemat('numpy_data/test.mat', {"label_cost": label_cost, "data_cost":data_cost, "smooth_cost":smooth_cost, "sparse":sparse, "cc_sparse_ns":cc_sparse_ns})
    if(should_solve_via_csv):
        np.savetxt("numpy_data/mishanius_labels.csv", label_cost, fmt='%i', delimiter=",")
        np.savetxt("numpy_data/data_cost.csv", data_cost, fmt='%i', delimiter=",")
        np.savetxt("numpy_data/smooth_cost.csv", smooth_cost, fmt='%i', delimiter=",")
    else:
        gc = gco.GCO()
        gc.create_general_graph(num_connected_componants, num_of_lines + 1, False)
        gc.set_smooth_cost(smooth_cost)
        gc.set_data_cost(np.array(np.int32(np.around(data_cost.copy()))))
        gc.set_label_cost(np.array([t for t in np.transpose(label_cost)]))
    for index, currEdge in enumerate(sparse):
        if (should_solve_via_csv):
            neighbors[index][0] = currEdge[0][0]
            neighbors[index][1] = currEdge[0][1]
            neighbors[index][2] = currEdge[1]
            np.savetxt("numpy_data/neighbors.csv", neighbors, fmt='%i', delimiter=",")
        else:
            gc.set_neighbor_pair(currEdge[0][0], currEdge[0][1], currEdge[1])
    if (solve_via_csv):
        return solve_via_csv(str(num_connected_componants), str(num_of_lines+1), str(len(neighbors)))
    else:
        try:
            gc.expansion()
            result_labels = gc.get_labels() + 1
            return result_labels
        except Exception as e:
            print("gco faild")
        finally:
            gc.destroy_graph()


def computeEdgeWeights(W):
    #   W = np.array(W).transpose()
    gcScale = 100
    beta = 1 / (0.5 * np.mean(W[W > 0]))
    edgeWeights = np.exp(-beta * W)
    edgeWeights[edgeWeights >= 1] = 0
    edgeWeights = np.round(gcScale * edgeWeights)
    return edgeWeights

def solve_via_csv(num_of_sites, num_oflabels, neighboor_pairs):
    process = Popen(['gco/testMain', num_of_sites, num_oflabels, neighboor_pairs], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    return np.array([int(i) for i in stdout.decode("utf-8").split(',') if i != ""])


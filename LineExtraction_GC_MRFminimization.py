import numpy as np
import gco
def LineExtraction_GC_MRFminimization(numLines, num, CCsparseNs, Dc, LabelCost):
    K = numLines + 1

    edgeWeights = computeEdgeWeights(CCsparseNs)

    gc = gco.GCO()
    gc.create_general_graph(num,K, False)

    threshHold = 10**7
    Dc[Dc > threshHold] = threshHold - 1
    #LabelCost[LabelCost > threshHold] = threshHold - 1
    #LabelCost = np.array(np.round(LabelCost), dtype=np.int32)

    sparse = []
    for rowIndex in range(num):
        for colIndex in range(num):
            if edgeWeights[rowIndex, colIndex] != 0 and 0 < rowIndex < colIndex < num:
                sparse.append(((rowIndex, colIndex), np.int32(edgeWeights[rowIndex, colIndex])))

    gc.set_data_cost(np.int32(np.round(Dc.transpose())))
    Smooth_cost = np.int32((np.ones(K) - np.eye(K)))
    gc.set_smooth_cost(Smooth_cost)
    #gc.set_label_cost(LabelCost)

    for currEdge in sparse:
        gc.set_neighbor_pair(currEdge[0][0], currEdge[0][1], currEdge[1])


    gc.expansion()
    Labels = gc.get_labels()
    gc.destroy_graph()
    return Labels



def computeEdgeWeights(W):
 #   W = np.array(W).transpose()
    gcScale = 100
    beta = 1/(0.5* np.mean(W[W > 0]))
    edgeWeights = np.exp(-beta * W)
    edgeWeights[edgeWeights >= 1] = 0
    edgeWeights = np.round(gcScale*edgeWeights)
    return edgeWeights

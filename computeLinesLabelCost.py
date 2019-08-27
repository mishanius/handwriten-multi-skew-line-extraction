import numpy as np
def computeLineLabelCost(L, Lines, numLines):
    acc = np.zeros((numLines + 1, 1))
    mask_ = L.flatten()
    L_ = Lines.flatten()

    for i in range(len(L_)):
        if (L_[i]) and mask_[i]:
            acc[int(L_[i])] +=  1

    LabelCost = np.exp(0.2*max(acc) / acc)
    LabelCost[numLines] = 0

    return  LabelCost
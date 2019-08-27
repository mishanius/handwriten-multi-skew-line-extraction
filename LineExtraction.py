from filterDocument import *
from scipy.ndimage import label as bwlabel
from BuildTreeLayer import BuildTreeLayer
import cv2
from approximateUsingPiecewiseLinear import approximateUsingPiecewiseLinear
def LineExtraction(I, scales):
    print("LineExtraction")
    _,_, max_response = filterDocument(I,scales)


    max_scale = max(scales)
    endThs = 50
    beginThs = findRootThreshold(max_response, endThs)
    Threshholds = np.arange(beginThs, endThs, 1)
    linesMask = TraverseTree(max_response, Threshholds, max_scale)


    return linesMask

def TraverseTree(max_response, Threshholds, max_scale):
    sz = max_response.shape
    res = np.zeros(sz, dtype=bool)
    markedChild = []
    npMax_respone = np.matrix(max_response)

    for ths in Threshholds:
        parentMask = npMax_respone > ths
        childMask = npMax_respone > ths + 1

        childL, childNum = bwlabel(childMask)
        parentL, parentNum = bwlabel(parentMask)
        links = BuildTreeLayer(parentL, parentNum, childL,childNum)

        marked = markedChild
        fitting = approximateUsingPiecewiseLinear(parentL, parentNum, marked, 3*max_scale)

        LineIndices = np.nonzero(fitting < 0.8 * max_scale)

        newIndices = np.setdiff1d(LineIndices, marked)
        marked = list(set(marked) | set(newIndices))
        markedChild = links
        markedChild = np.setdiff1d(markedChild, 0)



        res = np.bitwise_or(res, np.isin(parentL, newIndices))



        if marked == list(range(parentNum)):
            print("breaking")
            break


    return res


def findRootThreshold(max_response, my_max):

    my_min = np.matrix(max_response).min()
    print (my_min)
    interval = my_max-my_min
    i = interval/2+ my_min

    while interval > 0.5:
        mask = np.matrix(max_response) > i
        print (len(mask))
        _, num = bwlabel(mask)

        if num == 1:
            my_min = i
        else:
            my_max = i

        interval = my_max - my_min
        i = interval/2 + my_min

    begin_ths = i-0.5
    return begin_ths

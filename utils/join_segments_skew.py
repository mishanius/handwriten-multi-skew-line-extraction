import math
import numpy as np
from scipy.ndimage import label as bwlabel
from plantcv.plantcv import dilate
from skimage.morphology import reconstruction
import scipy.io as sio
from sklearn.decomposition import PCA
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors
import numpy.matlib
from scipy.sparse import csr_matrix
from scipy import ndimage

from permuteLabels import permuteLabels
from utils.approximate_using_piecewise_linear_pca import find_spline_with_numberofknots, \
    approximate_using_piecewise_linear_pca
from scipy.sparse.csgraph import connected_components
from numpy import linalg as LA
from scipy.sparse.csgraph import minimum_spanning_tree
from plantcv.plantcv.morphology import skeletonize, find_branch_pts

MATLAB_ROOT = "C:/Users/Itay/OneDrive - post.bgu.ac.il/academic/imageProcessing/LineExtraction2"

def join_segments_skew(L, newLines, newLinesNum, max_scale):

    num_of_knots = 20
    pca = PCA()
    # TODO deside if need transpose
    temp = regionprops(newLines)
    numOfKnots = 20
    end_points = np.zeros((newLinesNum, 8))
    for i in range(newLinesNum):
        try:
            pixel_list = temp[i].coords
            pca_res = pca.fit(pixel_list)
            pcav = pca_res.components_[0]
            theta = math.atan(pcav[1] / pcav[0])
            transformation = np.array([[math.cos(-theta), -math.sin(-theta)], [math.sin(-theta), math.cos(-theta)]])
            rotated_pixels = np.matmul(transformation, np.transpose(pixel_list))
        except Exception as e:
            continue

        x_rotated = rotated_pixels[0, :]
        y_rotated = rotated_pixels[1, :]
        # here is the slm-----------
        #     print(np.concatenate((np.transpose(x_endP),np.transpose(y_endP)),0))
        # here is the slm-----------
        x_endP = slm_knots[[3, 0, numOfKnots - 4, numOfKnots - 1]]
        y_endP = slm_coef[[3, 0, numOfKnots - 4, numOfKnots - 1]]
        r_inv = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        temp = np.matmul(r_inv, np.concatenate((np.transpose(x_endP), np.transpose(y_endP)), 0))
        end_points[i, :] = temp.flatten('F')

    external_ep = np.zeros((2 * newLinesNum, 2))
    external_ep[0:newLinesNum, :] = end_points[:, 2:4]
    external_ep[newLinesNum:2 * newLinesNum, :] = end_points[:, 6:8]

    K = min(4, 2 * newLinesNum - 1)
    nbrs = NearestNeighbors(K + 1).fit(external_ep)
    dist, indexs = nbrs.kneighbors(external_ep)
    indexs = indexs[:, 1:]
    dist = dist[:, 1:]
    dist_mat = np.full((2 * newLinesNum, K), np.inf)

    for j in range(K):
        for i in range(2 * newLinesNum):

            lhsIdx = i % newLinesNum
            #         print("lhsIdx:{}\n".format(lhsIdx))
            # if lhsIdx == 0:
            #     lhsIdx = newLinesNum - 1
            rhsIdx = indexs[i, j] % newLinesNum
            # if rhsIdx == 0:
            #     rhsIdx = newLinesNum - 1
            if lhsIdx == rhsIdx:
                dist_mat[i, j] = np.inf
            else:
                if indexs[i, j] > newLinesNum - 1:
                    rhsOuterIndices = [6, 7]
                    rhsInnerIndices = [4, 5]
                else:
                    rhsOuterIndices = [2, 3]
                    rhsInnerIndices = [0, 1]
                if i >= newLinesNum:
                    lhsOuterIndices = [6, 7]
                    lhsInnerIndices = [4, 5]
                else:
                    lhsOuterIndices = [2, 3]
                    lhsInnerIndices = [0, 1]

                #             print("lhsIdx:{}\n".format(lhsIdx))
                #             print(end_points.shape)
                l = LA.norm(end_points[lhsIdx, lhsOuterIndices] - end_points[lhsIdx, lhsInnerIndices]) + \
                    LA.norm(end_points[rhsIdx, rhsOuterIndices] - end_points[rhsIdx, rhsInnerIndices]) + \
                    LA.norm(end_points[rhsIdx, rhsOuterIndices] - end_points[lhsIdx, lhsOuterIndices])
                l1 = LA.norm(end_points[rhsIdx, rhsInnerIndices] - end_points[lhsIdx, lhsInnerIndices])
                dist_mat[i, j] = l / l1
    print("DistMatrix:{}\n".format(dist_mat))
    dist_mat = np.exp(15 * (dist_mat - 1))
    tempMat = np.full((2 * newLinesNum, 2 * newLinesNum), np.inf)
    for i in range(2 * newLinesNum):
        j = np.argmin(dist_mat[i, :])
        tempMat[i, indexs[i, j]] = dist_mat[i, j]
        tempMat[indexs[i, j], i] = dist_mat[i, j]

    for i in range(newLinesNum):
        tempMat[i, i + newLinesNum] = 0
        tempMat[i + newLinesNum, i] = 0

    cnt = sum(sum(tempMat != np.inf))
    temp_to_csv = np.vstack(np.sort(np.argwhere(tempMat != np.inf)[:, 1]))
    E = np.full(((cnt + 4 * newLinesNum), 2), -1)
    w = np.zeros(((cnt + 4 * newLinesNum), 1))

    idx = 0
    for j in range(2 * newLinesNum):
        for i in range(2 * newLinesNum):
            if tempMat[i, j] != np.inf:
                E[idx, 0: 2] = [i, j]
                w[idx] = tempMat[i, j]
                idx = idx + 1

    DensityLabelCost = extractDensity(L, newLines, newLinesNum)
    DensityLabelCost[DensityLabelCost > 10000] = 10000

    n = 2 * newLinesNum
    E[cnt: cnt + n, 0] = n
    E[cnt: cnt + n, 1] = range(n)
    E[cnt + n: cnt + 2 * n, 0] = range(n)
    E[cnt + n: cnt + 2 * n, 1] = n
    w[cnt: cnt + 2 * n] = np.matlib.repmat(DensityLabelCost, 4, 1)
    mw = w + 1

    A = csr_matrix((w.flatten(), (E[:, 0], E[:, 1])), (n + 1, n + 1))
    M = csr_matrix((mw.flatten(), (E[:, 0], E[:, 1])), (n + 1, n + 1))

    As = csr_matrix((np.full((E.shape[0], 1), True).flatten(), (E[:, 0], E[:, 1])), (n + 1, n + 1))

    i_j = np.argwhere(np.transpose(As))
    i = i_j[:, 0]
    j = i_j[:, 1]

    Ai = csr_matrix((np.arange(1, E.shape[0] + 1), (i, j)), (n + 1, n + 1))

    mult_res = Ai.multiply(A != 0)
    ax, ay = mult_res.nonzero()
    non_zeros = mult_res[ax, ay]

    non_zeros_x, non_zeros_y = A.nonzero()

    v = np.asarray(np.zeros((As.nnz, 1))).flatten()
    v[non_zeros - 1] = A[non_zeros_x, non_zeros_y]

    Tcsr = minimum_spanning_tree(M)
    wx, wy = Tcsr.nonzero()
    newE = np.array([wx, wy])
    rows = np.argwhere(newE == n)[:, 1]
    mask = np.ones(newE.shape)
    mask[rows] = False
    newE = newE[mask]

    newA = np.asarray((np.ones((newE.shape[0], 1)), (newE[:, 0], newE[:, 1])), (n, n))
    newA = newA + np.transpose(newA)
    n_components, ci = connected_components(csgraph=newA, directed=False, return_labels=True)
    new_labeling = ci[:newLinesNum]
    LUT = np.zeros((1,65536), np.uint16)
    LUT[1:len(new_labeling) + 1] = np.transpose(new_labeling)
    combinedLines = np.double(LUT[newLines])
    new_segments = np.zeros(newLines.shape)
    newSegmentsCnt = 1

    for i in range(len(newE)):
        lhsIdx = newE[i, 0] % newLinesNum
        rhsIdx = newE[i, 1] % newLinesNum
        if lhsIdx == rhsIdx:
            continue
        else:
            [mask, LabelNum] = join_segments_helper(L, newLines, combinedLines, [lhsIdx, rhsIdx], max_scale, temp)
            if LabelNum:
                new_segments[mask] = newSegmentsCnt
                combinedLines[mask] = LabelNum
                newSegmentsCnt = newSegmentsCnt + 1

    cnt = max(np.amax(combinedLines))
    for i in range(cnt):
        [L, num] = bwlabel(combinedLines == i)
        if (num > 1):
            for j in range(num):
                combinedLines[L == j] = cnt + 1
            cnt = cnt + 1

    combinedLines,_ = permuteLabels(combinedLines)


def join_segments_helper(L, Lines, combinedLines, segments2Join, max_scale, pixelList):

    mask = Lines
    length = len(segments2Join)
    LabelNum = False
    minX = np.inf
    minY = np.inf
    maxX = 0
    maxY = 0
    for i in range(len(segments2Join)):
        X = pixelList[segments2Join[i]].coords
        minX = min(np.amax(X[:, 0]), minX)
        minY = min(np.amax(X[:, 1]), minY)
        maxX = max(np.amax(X[:, 0]), maxX)
        maxY = max(np.amax(X[:, 1]), maxY)

    CroppedLines = Lines[minY:maxY, minX: maxX]

    for i in range(length - 1):
        bw1 = CroppedLines == segments2Join[i]
        bw2 = CroppedLines == segments2Join[i + 1]
        #TODO talk to baret, no quasi-euclidean transformation avaliable https://www.mathworks.com/help/images/ref/bwdist.html
        D1 = ndimage.distance_transform_edt(bw1==0)
        D2 = ndimage.distance_transform_edt(bw2==0)
        D = np.round((D1 + D2) * 32) / 32


        lm = ndimage.minimum_filter(D, np.ones((3,3)))
        paths = (D == lm)
        #TODO understand the np.inf | paths = bwmorph(paths, 'skel', np.inf)
        paths = skeletonize(paths)

        tempMask = dilate(paths, ksize=10, i=1)

        mask = np.full(L.shape, False)

        mask[minY: maxY, minX: maxX] = tempMask
        AdjacentIndices = np.unique(combinedLines[mask])

        rows_to_remove = np.argwhere[AdjacentIndices == 0]
        mask = np.ones(AdjacentIndices.shape)
        mask[rows_to_remove] = False
        AdjacentIndices = rows_to_remove[mask]

        if (len(AdjacentIndices) > 1) or (not np.any(np.logical_and(mask , L))):
            LabelNum = False
        else:
            mask = np.logical_and(mask ,np.logical_not(combinedLines))
            bw1 = Lines == segments2Join[i]
            bw2 = Lines == segments2Join[i + 1]
            # TODO previously no dialation
            t = reconstruction(np.logical_or(bw1, bw2), np.logical_or(combinedLines == AdjacentIndices , mask), method='dilation')
            fitting = approximate_using_piecewise_linear_pca(t, 1, 0, 0)
            if fitting[0] < 0.8 * max_scale:
                LabelNum = AdjacentIndices
            else:
                LabelNum = False
    return mask, LabelNum



def extractDensity(L, Lines, numLines):
    acc = np.zeros((numLines + 1, 1))
    mask_ = L.flatten()
    L_ = Lines.flatten()

    for i in range(len(L_)):
        if L_[i] and mask_[i]:
            acc[L_[i]] = acc[L_[i]] + 1
    acc = acc[1:]
    acc = np.exp(0.2 * np.amax(acc) / acc)
    acc[acc < 6.5] = 6.5
    return acc

def prepare_for_debug_join_segments_helper():
    l = sio.loadmat("C:/Users/Itay/OneDrive - post.bgu.ac.il/academic/imageProcessing/LineExtraction2/L.mat")
    L = l['L']
    Lines = sio.loadmat(
        "C:/Users/Itay/OneDrive - post.bgu.ac.il/academic/imageProcessing/LineExtraction2/Lines.mat")
    Lines = Lines['Lines']
    combinedLines = sio.loadmat(
        "C:/Users/Itay/OneDrive - post.bgu.ac.il/academic/imageProcessing/LineExtraction2/combinedLines.mat")
    combinedLines = combinedLines['combinedLines']
    segments2Join = sio.loadmat(
        "C:/Users/Itay/OneDrive - post.bgu.ac.il/academic/imageProcessing/LineExtraction2/segments2Join.mat")
    segments2Join = segments2Join['segments2Join'][0]
    max_scale = sio.loadmat(
        "C:/Users/Itay/OneDrive - post.bgu.ac.il/academic/imageProcessing/LineExtraction2/max_scale.mat")
    max_scale = max_scale['max_scale'][0][0]
    new_lines = sio.loadmat(
        "C:/Users/Itay/OneDrive - post.bgu.ac.il/academic/imageProcessing/LineExtraction2/newLines.mat")
    new_lines = new_lines['newLines']
    new_lines = sio.loadmat(
        "C:/Users/Itay/OneDrive - post.bgu.ac.il/academic/imageProcessing/LineExtraction2/newLines.mat")
    new_lines = new_lines['newLines']
    pixel_lists = regionprops(new_lines)
    join_segments_helper(L, Lines, combinedLines, segments2Join-1, max_scale, pixel_lists)

def prepare_for_debug_join_segments():
    l = sio.loadmat("{}/{}".format(MATLAB_ROOT, "L.mat"))
    L = l['L']
    Lines = sio.loadmat(
        "{}/{}".format(MATLAB_ROOT, "Lines.mat"))
    Lines = Lines['Lines']
    combinedLines = sio.loadmat(
        "{}/{}".format(MATLAB_ROOT, "combinedLines.mat"))
    combinedLines = combinedLines['combinedLines']
    segments2Join = sio.loadmat(
        "{}/{}".format(MATLAB_ROOT, "segments2Join.mat"))
    segments2Join = segments2Join['segments2Join'][0]
    max_scale = sio.loadmat(
        "{}/{}".format(MATLAB_ROOT, "max_scale.mat"))
    max_scale = max_scale['max_scale'][0][0]
    new_lines = sio.loadmat(
        "{}/{}".format(MATLAB_ROOT, "newLines.mat"))
    new_lines = new_lines['newLines']
    new_lines = sio.loadmat(
        "{}/{}".format(MATLAB_ROOT, "newLines.mat"))
    new_lines = new_lines['newLines']
    pixel_lists = regionprops(new_lines)
    return [L, Lines, combinedLines, segments2Join, max_scale, pixel_lists]

def prepare_for_debug_join_segments():
    l = sio.loadmat("{}/{}".format(MATLAB_ROOT, "L.mat"))
    L = l['L']
    new_lines = sio.loadmat(
        "{}/{}".format(MATLAB_ROOT, "newLines.mat"))
    new_lines = new_lines['newLines']
    newLinesNum = sio.loadmat(
        "{}/{}".format(MATLAB_ROOT, "newLinesNum.mat"))
    newLinesNum = newLinesNum['newLinesNum'][0][0]
    max_scale = sio.loadmat(
        "{}/{}".format(MATLAB_ROOT, "max_scale.mat"))
    end_points = sio.loadmat(
        "{}/{}".format(MATLAB_ROOT, "endPoints.mat"))
    end_points = end_points['endPoints']
    max_scale = max_scale['max_scale'][0][0]
    slm = sio.loadmat("{}/{}".format(MATLAB_ROOT, "slm.mat"))
    slm_knots = slm['knots']
    slm_coef = slm['coef']
    join_segments_skew()

if __name__ == "__main__":
    prepare_for_debug_join_segments_helper()



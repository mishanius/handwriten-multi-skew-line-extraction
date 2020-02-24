import  numpy as np
from skimage import img_as_bool, io, color, morphology
from scipy.ndimage import label as bwlabel
from skimage.morphology import reconstruction
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors
def RefineBinnaryOverlappingComponents(CCsl, CCsNum, linesL, linesNum):
    sz = CCsl.shape
    result = np.zeros(sz)

    res = np.zeros((CCsNum + 1, linesNum))
    CCsLF = np.array(CCsl).flatten()
    linesLF = np.array(linesL).flatten()

    for i in range(len(CCsLF)):
        if CCsLF[i] and linesLF[i]:
            res[CCsLF[i]-1, int(linesLF[i]-1)] = 1

    temp = np.sum(res, 1)
    CCindices = np.nonzero(temp > 1)
    skel = morphology.medial_axis(linesL)

    for i in range(len(CCindices)):
        idx = CCindices[i]
        cc = CCsl == idx
        linesIndices = np.nonzero(res[idx])

        if len(linesIndices) == 2:
            skelLabels, _ = bwlabel(np.bitwise_and(skel, np.isin(linesL, linesIndices)), np.ones((3,3)))
            temp = reconstruction(np.bitwise_and(cc, skelLabels), skelLabels > 0)
            _, num = bwlabel(temp, np.ones((3,3)))
            if num < 2:
                continue

        ccPixels = regionprops(cc)
        ccPixelsList = []

        for lst in ccPixels:
            ccPixelsList.append(lst.coords)

        Dist = np.zeros((len(ccPixelsList), len(linesIndices)))

        for j in range(len(linesIndices)):
            line = linesL == linesIndices[j]
            linePixel = regionprops(line)
            linePixelList = []
            for lst in linePixel:
                linePixelList.append(lst.coords)

            nbrs = NearestNeighbors(2).fit(linePixelList)
            Dist[:, j], indexes = nbrs.kneighbors(ccPixelsList)

        loc = np.argmin(Dist)
        PixelIdxList = ccPixelsList

        for j in range(len(linesIndices)):
            indices = loc ==j
            result[PixelIdxList[indices]] = linesIndices[j]

    return result



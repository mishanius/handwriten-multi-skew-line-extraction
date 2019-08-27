from permuteLabels import  permuteLabels
from scipy.ndimage import label as bwlabel
from computeNsSystem import  computeNsSystem
from computeLinesDC import  computeLinesDC
from computeLinesLabelCost import  computeLineLabelCost
import numpy as np
from RefineBinnaryOverlappingComponents import RefineBinnaryOverlappingComponents
from LineExtraction_GC_MRFminimization import LineExtraction_GC_MRFminimization
def PostProcessByMRF(L, num, linesMask, charRange):

    CCsparseNs = computeNsSystem(L, num)
    [result, Labels, numLines] = PostProcessByMRFHelper(L, num, linesMask, CCsparseNs, charRange)
    return [result, Labels, numLines]



def PostProcessByMRFHelper(L, num, LineMask, CCsparseNs, charRange):
    LineLabels, n_label = bwlabel(LineMask.__invert__())
    Lines, numLines = permuteLabels(LineLabels)
    Dc = computeLinesDC(Lines, numLines, L, num, charRange[1])
    LabelCost =  computeLineLabelCost(L, Lines, numLines)
    Labels = LineExtraction_GC_MRFminimization(numLines, num, CCsparseNs, Dc, LabelCost)
    Labels[Labels == numLines + 1] = 0
    residualLines = np.isin(Lines, Labels)
    Lines[residualLines.__invert__()] = 0
    result = drawLabels(L, Labels)
   # RefinedCCs = RefineBinnaryOverlappingComponents(L, num, Lines, numLines)

    #tempMask = RefinedCCs >= 0

    #result[tempMask] = RefinedCCs[tempMask]
    return [result, Labels, numLines]



def drawLabels(L, Labels):
    L = np.uint16(L)
    LUT = np.zeros(65536, np.uint16)
    LUT[1:len(Labels)+1] = Labels
    result = np.double(LUT[L])
    return result


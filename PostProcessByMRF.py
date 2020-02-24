from scipy.ndimage import label as bwlabel
from computeNsSystem import computeNsSystem
from computeLinesDC import computeLinesDC, compute_lines_data_cost
from computeLinesLabelCost import computeLineLabelCost
import numpy as np
from LineExtraction_GC_MRFminimization import LineExtraction_GC_MRFminimization, line_extraction_GC

# @partial_image(2, "post_process_by_mfr")
def post_process_by_mfr(labeled_raw_components, raw_components_num, lines_mask, char_range):
    cc_sparse_ns = computeNsSystem(labeled_raw_components, raw_components_num)
    [result, labels, num_lines] = post_process_by_mfr_helper(labeled_raw_components,
                                                             raw_components_num,
                                                             lines_mask,
                                                             char_range,
                                                             cc_sparse_ns)
    return [result, labels, num_lines]


def post_process_by_mfr_helper(labeled_raw_components, raw_components_num, line_mask, char_range, cc_sparse_ns=None):
    # line mask should be binarized already
    line_mask_lables, n_masked_lines = bwlabel(line_mask, np.ones((3,3)))
    data_cost = compute_lines_data_cost(line_mask_lables, n_masked_lines, labeled_raw_components, raw_components_num,
                                        char_range[1])

    label_cost = computeLineLabelCost(labeled_raw_components, line_mask_lables, n_masked_lines)

    labels = line_extraction_GC(raw_components_num, n_masked_lines, data_cost, cc_sparse_ns, label_cost)

    result = drawLabels(labeled_raw_components, labels)
    return [result, labels, n_masked_lines]


# L = original image
def PostProcessByMRFHelper(L, num, LineMask, CCsparseNs, charRange):
    # LineLabels, n_label = bwlabel(LineMask.__invert__())
    LineLabels, n_label = bwlabel(LineMask)
    print("got {} labels".format(n_label))
    Lines, numLines = LineLabels, n_label
    # Lines, numLines = permuteLabels(LineLabels)
    Dc = computeLinesDC(Lines, numLines, L, num, charRange[1])
    LabelCost = computeLineLabelCost(L, Lines, numLines)
    Labels = LineExtraction_GC_MRFminimization(numLines, num, CCsparseNs, Dc, LabelCost)
    Labels[Labels == numLines + 1] = 0
    # residualLines = np.isin(Lines, Labels)
    # Lines[residualLines.__invert__()] = 0
    result = drawLabels(L, Labels)
    # RefinedCCs = RefineBinnaryOverlappingComponents(L, num, Lines, numLines)

    # tempMask = RefinedCCs >= 0

    # result[tempMask] = RefinedCCs[tempMask]
    return [result, Labels, numLines]


def drawLabels(L, Labels):
    L = np.uint16(L)
    LUT = np.zeros(65536, np.uint16)
    LUT[1:len(Labels) + 1] = Labels
    result = np.double(LUT[L])
    return result

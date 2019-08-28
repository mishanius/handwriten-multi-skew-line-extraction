import numpy as np
import cv2
from estimateBinaryHeight import *
from scipy.ndimage import label as bwlabel
from LineExtraction import *
from PostProcessByMRF import  *
from matplotlib import pylab as pt
from skimage.color import label2rgb
# Load an color image in grayscale
I = cv2.imread('binary_hetero_doc.png', 0)
bin = cv2.bitwise_not(I)



charRange = estimateBinaryHeight(bin)


LineMask = LineExtraction(I, charRange)

# t = LineMaskInv.astype(np.int8) * 100

pt.imshow(LineMask)
pt.show()
L, num = bwlabel(bin)
[result, Labels, newLines] = PostProcessByMRF(L, num, LineMask, charRange)


r = label2rgb(result)

cv2.imshow('image', r)
cv2.waitKey(0)
cv2.destroyAllWindows()
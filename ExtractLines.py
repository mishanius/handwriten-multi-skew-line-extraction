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


# LineMask = LineExtraction(I, charRange)
# pt.imsave("images/mask.png",LineMask)
# LineMask = np.logical_not(LineMask)
# np.save("numpy_data/LineMask",LineMask)
LineMask = np.load('numpy_data/LineMask.npy')
# LineMask = np.logical_not(LineMask)
# LineMask = cv2.imread("images/mask.png", 0)
# LineMask = cv2.bitwise_not(LineMask)
L, num = bwlabel(bin)
[result, Labels, newLines] =post_process_by_mfr(L, num, LineMask, charRange)


r = label2rgb(result, bg_color=(0,0,0))
# new_image = r.astype(np.uint8)
# new_image_red, new_image_green, new_image_blue = new_image
# new_rgb = np.dstack([new_image_red, new_image_green, new_image_blue])
cv2.imshow('image', r)
cv2.waitKey(0)
cv2.destroyAllWindows()

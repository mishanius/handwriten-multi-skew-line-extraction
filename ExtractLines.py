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
# LineMask2 = LineMask.astype(int)
# LineMask2[LineMask==False] = 0
# LineMask2[LineMask==True] = 255
# new_image = LineMask2.astype(np.uint8)
# # new_image_red, new_image_green, new_image_blue = new_image
# # new_rgb = np.dstack([new_image_red, new_image_green, new_image_blue])
# cv2.imshow("WindowNameHere", new_image)
# cv2.waitKey(0)
# # t = LineMaskInv.astype(np.int8) * 100
# cv2.imwrite('image.png', new_image)
# pt.imsave("images/mask.png",LineMask)
# LineMask =pt.imread("images/mask.png",0)
# LineMask = cv2.imread('images/image_mask.png', 0)
# LineMask = cv2.bitwise_not(LineMask)
# pt.imshow(LineMask)
# pt.show()
L, num = bwlabel(bin)
# [result, Labels, newLines] = PostProcessByMRF(L, num, LineMask, charRange)
[result, Labels, newLines] =post_process_by_mfr(L, num, LineMask, charRange)


r = label2rgb(result, bg_color=(0,0,0))
# new_image = r.astype(np.uint8)
# new_image_red, new_image_green, new_image_blue = new_image
# new_rgb = np.dstack([new_image_red, new_image_green, new_image_blue])
cv2.imshow('image', r)
cv2.waitKey(0)
cv2.destroyAllWindows()
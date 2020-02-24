import numpy as np
import cv2
import matplotlib.pyplot as plt

import scipy.io as sio

MATLAB_ROOT = "C:/Users/Itay/OneDrive - post.bgu.ac.il/academic/imageProcessing/LineExtraction2"

def draw_labels(L, Labels):
    labels2 = sio.loadmat(
        "{}/{}".format(MATLAB_ROOT, "Labels.mat"))
    labels2 = labels2['Labels']
    print(np.array_equal(labels2.reshape((labels2.shape[0],)), np.array(Labels)))
    L = np.uint16(L)
    LUT = np.zeros(65536, np.uint16)
    LUT[1:len(Labels) + 1] = Labels
    result = np.double(LUT[L])
    return result

def gen_lut():
  """
  Generate a label colormap compatible with opencv lookup table, based on
  Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
  appendix C2 `Pseudocolor Generation`.
  :Returns:
    color_lut : opencv compatible color lookup table
  """
  tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
  arr = np.arange(256)
  r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
  g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
  b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
  return np.concatenate([[[r]], [[g]], [[b]]]).T

def labels2rgb(labels):
  """
  Convert a label image to an rgb image using a lookup table
  :Parameters:
    labels : an image of type np.uint8 2D array
    lut : a lookup table of shape (256, 3) and type np.uint8
  :Returns:
    colorized_labels : a colorized label image
  """
  lut = gen_lut()
  return cv2.LUT(cv2.merge((labels, labels, labels)), lut)

if __name__ == '__main__':
  labels = np.random.randint(1, size=(10, 10)).astype(np.uint8)
  lut = gen_lut()
  # rgb = labels2rgb(labels, lut)
  # plt.imshow(rgb)
  # plt.title('fskel')
  # plt.show()
  # # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
  # # cv2.resizeWindow('image', 600, 600)
  # cv2.imshow('image', rgb)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
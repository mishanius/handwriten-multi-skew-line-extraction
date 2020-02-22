from estimateBinaryHeight import *
from LineExtraction import *
from PostProcessByMRF import *
from matplotlib import pylab as pt
from skimage.color import label2rgb
import argparse


def extract_lines(image_path, mask_path=None):
    # Load a color image in grayscale
    I = cv2.imread(image_path, 0)
    bin = cv2.bitwise_not(I)
    charRange = estimateBinaryHeight(bin)
    if mask_path is None:
        LineMask = LineExtraction(I, charRange)
        pt.imsave("images/mask.png", LineMask)
        LineMask = np.logical_not(LineMask)
    else:
        LineMask = cv2.imread(mask_path, 0)
        LineMask = cv2.bitwise_not(LineMask)

    L, num = bwlabel(bin)
    [result, Labels, newLines] = post_process_by_mfr(L, num, LineMask, charRange)
    r = label2rgb(result, bg_color=(0, 0, 0))
    cv2.imshow('image', r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract lines from doc')
    parser.add_argument('--image_path',  type=str, default='binary_hetero_doc.png', required=False,
                        help='path to the doc image')
    parser.add_argument('--mask_path', type=str, default=None, required=False,
                        help='path for already created mask for example images/image_mask.png')
    args = parser.parse_args()
    extract_lines(args.image_path, args.mask_path)

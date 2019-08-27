from scipy.ndimage import label as bwlabel
from skimage.measure import regionprops
import statistics

def estimateBinaryHeight(bin):
    print("estimateBinaryHeight")
    L,_ = bwlabel(bin)

    props = regionprops(L)
    Height =[]
    for prop in props:
       Height.append(prop.bbox[2]-prop.bbox[0])
    
    mu = statistics.mean(Height)
    sigma = statistics.stdev(Height)
    lower = (mu)/2
    upper = (mu+sigma/2)/2

    return [lower, upper]

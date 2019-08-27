from skimage.measure import regionprops
def extractCentroids(L):
    temp = regionprops(L.transpose())
    X = []
    for prop in temp:
        X.append([prop.centroid[0], prop.centroid[1]])
    return X

import numpy as np
from scipy import optimize
from skimage.measure import regionprops
from numpy.linalg import norm
import matplotlib.pyplot as plt
def approximateUsingPiecewiseLinear(L,num, marked, ths):

    print("num is {}".format(num))
    res = regionprops(L)
    numOfKnots = 20
    fitting = np.zeros((num, numOfKnots - 1))

    for i in range(num):
        # if i in marked:
        #     fitting[i] = np.zeros(len(fitting[i]))
        #     continue

        pixelList = res[i].coords
        x = column(pixelList, 0)
        y = column(pixelList, 1)
        #plt.plot(y,x)
        #plt.show()
        try:
            p=np.polyfit(y, x, 1)
        except:
            continue
        y_hat = np.polyval(p, y)

        fit = norm(y_hat-x, 1) / len(y)
        print("first fit is {}".format(fit))

        fitting[i] = np.unique(fit)
        print(fitting)

    res = []
    for row in fitting:
            res.append(row[0])
    return np.array(res)

def column(matrix, i):

    return [row[i] for row in matrix]


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])


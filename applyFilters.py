import numpy as np
from ani import anigauss
def applyFilters(inImage,sz, scale,theta = 0,eta =3):

        response = anigauss(inImage,scale, eta*scale, theta,2,0)
        res = (np.ones((sz[0],sz[1]), dtype=int)).tolist()
        return [res, response]
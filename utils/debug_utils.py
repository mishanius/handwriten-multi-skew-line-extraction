import matplotlib.pyplot as plt


import scipy.io as sio
MATLAB_ROOT = "C:/Users/Itay/OneDrive - post.bgu.ac.il/academic/imageProcessing/LineExtraction2"

def debug_plot_image(im, name):
    cm = plt.get_cmap('gray')
    kw = {'cmap': cm, 'interpolation': 'none', 'origin': 'upper'}
    plt.imshow(im, **kw)
    plt.title(name)
    plt.show()

def load_from_matlab(name_of_var):
    var = sio.loadmat("{}/{}".format(MATLAB_ROOT, "{}.mat".format(name_of_var)))
    var = var['{}'.format(name_of_var)]
    return var



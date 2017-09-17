import numpy as np
import tensorflow as tf
import h5py
from scipy.misc import imresize
import matplotlib.pyplot as plt
from PIL import Image

#Graphical
def show(img):
    fig = plt.figure()
    plt.imshow(img, cmap='Greys_r')
    plt.show()

def showLoss(loss_data, smoothing = 100):
    fig = plt.figure()
    hamming = smooth(loss_data, smoothing, 'hamming')
    iters = loss_data.shape[0]
    plt.plot(xrange(iters), loss_data, c='grey')
    plt.plot(xrange(hamming.shape[0]), hamming, c='r')

def showMultiLoss(loss_data, p1, p2, smoothing = 100):
    fig = plt.figure()
    hamming = smooth(loss_data, smoothing, 'hamming')
    iters = loss_data.shape[0]

    p1_max = smooth(p1, smoothing, 'hamming')
    p2_max = smooth(p2, smoothing, 'hamming')

    plt.plot(xrange(p1_max.shape[0]), p1_max, color = 'orange')
    plt.plot(xrange(p2_max.shape[0]), p2_max, color = 'wheat')
    plt.plot(xrange(iters), loss_data, c='grey')
    plt.plot(xrange(hamming.shape[0]), hamming, c='r')


def save(image, name='out', normalize= True):

    if normalize and not image.mean()==0:
        image = image+np.abs(image.min())
        image = 255*(image/image.max())
        image = np.squeeze(image)
    #print(im.shape)
    result = Image.fromarray(image.astype(np.uint8))
    result.save(name+'.jpg')

def usual_save(image, name='out'):
    result = Image.fromarray(image.astype(np.uint8))
    result.save(name+'.jpg')

def xcsurface(xc):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    N=xc.shape[0]
    M=xc.shape[1]
    X = np.arange(0, N, 1)
    Y = np.arange(0, M, 1)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure("xc") #,figsize=(10,10))
    plt.clf()

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, xc, rstride=10, edgecolors="k",
                    cstride=10, cmap=cm.copper, alpha=1, linewidth=0,
                    antialiased=False)
    ax.set_zlim(-0.5, 2)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=10)

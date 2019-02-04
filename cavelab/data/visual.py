import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import h5py
from scipy.misc import imresize

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io


#Graphical
def show(img):
    fig = plt.figure()
    plt.imshow(img, cmap='Greys_r')
    plt.show()

def draw_curve(x, y, name='curve'):
    fig = plt.figure()
    plt.plot(x, y, c='grey')
    plt.savefig()

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

def normalize(image, normalize=True):
    if not image.mean()==0:
        image = image+np.abs(image.min())
        image = 255*(image/image.max())
        image = np.squeeze(image)
    return image

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

# move to visualize
def flow(_flow):

    hsv = np.zeros((_flow.shape[0], _flow.shape[1], 3), dtype=np.float32)
    hsv[...,1] = 255

    mag, ang = cv2.cartToPolar(_flow[...,0], _flow[...,1])
    hsv[...,0] = ang*180/np.pi
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    grid = draw_vector_field(ang, hsv)
    return rgb, grid

def draw_vector_field(ang, hsv, my_dpi=160, width=312):
    if hsv.shape[1]<32:
        rate = 1
    else:
        rate = int(10*hsv.shape[1]/256)

    plt.figure(figsize=(width/my_dpi, width/my_dpi), dpi=my_dpi, frameon=False)
    X, Y = np.meshgrid(np.arange(hsv.shape[0])[::rate],
                       np.arange(hsv.shape[1])[::rate])

    U = np.cos(ang[::rate,::rate]) * hsv[...,2][::rate,::rate]
    V = np.sin(ang[::-rate,::-rate]) * hsv[...,2][::-rate,::-rate]

    Q = plt.quiver(X, Y, U, V)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    crop = int(0.1*width)
    fig = np.array(Image.open(buf))[crop+8:width-crop,crop+8:width-crop,0:3]
    plt.close('all')

    return fig

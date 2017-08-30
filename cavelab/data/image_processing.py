from scipy.misc import imresize
from scipy import ndimage
import numpy as np
from math import floor

def resize(image, ratio=(1/3.0, 1/3.0), order=0):
    return ndimage.interpolation.zoom(image, ratio, order=order)/255.0

def normalize(image):
    if len(image.shape)>2:
        for i in range(image.shape[2]):
            image[:,:,i] += abs(image[:,:,i].min())
            image[:,:,i] /= abs(image[:,:,i].max())
            image[:,:,i] *= 255.0
    else:
        image += abs(image.min())
        image /= abs(image.max())
        image *= 255

    return image

def central_crop(images, size):
    shape = images.shape
    return images[:, shape[1]/2-size[1]/2:shape[1]/2+size[1]/2, shape[2]/2-size[2]/2:shape[2]/2+size[2]/2]

def random_crop(images, size):
    offset = [int(floor((images.shape[i]-size[i])*np.random.random())) for i in range(len(images.shape))]
    return images[offset[0]:offset[0]+size[0], offset[1]:offset[1]+size[1],offset[2]:offset[2]+size[2]], offset

def f(x, y, pad, length):
    if (x>=pad and y>=pad) and (x<length-pad and y<length-pad):
        return 1

    #corners
    scale_down = 4*float(pad)
    if (x<pad and y<pad):
        return (x+y+1)/(scale_down)

    if (x>length-pad-1 and y>length-pad-1):
        return (2*length-x-y-1)/(scale_down)  #(length-x-1)/(2*float(pad)**2)

    if (x<pad and y>length-pad-1):
        return (length+x-y)/(scale_down) #y/(2*float(pad)**2)

    if (x>length-pad-1 and y<pad):
        return (length-x+y)/(scale_down)

    #edges
    if (x<pad) and (y>=pad and y<=length-pad-1):
        return (x+0.5)/float(pad)

    if (y<pad) and (x>=pad and x<=length-pad-1):
        return (y+0.5)/float(pad)

    if (x>pad) and (y>=pad and y<=length-pad-1):
        return (length-x-0.5)/float(pad)

    if (y>pad) and (x>=pad and x<=length-pad-1):
        return (length-y-0.5)/float(pad)

    return 0

def get_blend_map(pad, size):
    blend_map = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            blend_map[x,y] = f(x,y, pad, size)
    return blend_map

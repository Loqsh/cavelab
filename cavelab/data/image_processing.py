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


'''
    reads data from matrix
    if borders are passed then instead of error return black

'''
def read_without_borders_2d(data, (x,y), (off_x, off_y), scale_ratio=(1,1)):
    shape = data.shape
    crop = [x,x+off_x, y, y+off_y]
    (padd_x, padd_x_size, padd_y, padd_y_size) = (0,off_x,0,off_y)

    if crop[0]<0:
        crop[0]=0
        padd_x = abs(x)

    if crop[2]<0:
        crop[2]=0
        padd_y = abs(y)

    if crop[1]>shape[0]:
        padd_x_size = off_x-abs((shape[0]-crop[1]))
        crop[1] = shape[0]

    if crop[3]>shape[1]:
        padd_y_size = off_y-abs((shape[1]-crop[3]))
        crop[3] = shape[1]

    sample = np.zeros((off_x,off_y))

    if padd_x>=0 and padd_x<off_x and padd_y>=0 and padd_y<off_y and padd_x_size>0 and padd_x_size<=off_x and padd_y_size>0 and padd_y_size<=off_y:
        sample[padd_x:padd_x_size, padd_y:padd_y_size] = np.array(data[crop[0]:crop[1], crop[2]:crop[3]])[:,:]

    return resize(sample, ratio=scale_ratio)


def read_without_borders_3d(data, (x,y,z), (off_x,off_y,off_z), scale_ratio=(1,1,1)):
    shape = data.shape
    crop = [x,x+off_x, y, y+off_y, z, z+off_z]
    (padd_x, padd_x_size, padd_y, padd_y_size, padd_z, padd_z_size) = (0,off_x,0,off_y,0,off_z)
    print('read_without_borders_3d')
    print(shape)
    print(crop)
    if crop[0]<0:
        crop[0]=0
        padd_x = abs(x)

    if crop[2]<0:
        crop[2]=0
        padd_y = abs(y)

    if crop[4]<0:
        crop[4]=0
        padd_z = abs(z)

    if crop[1]>shape[0]:
        padd_x_size = off_x-abs((shape[0]-crop[1]))
        crop[1] = shape[0]

    if crop[3]>shape[1]:
        padd_y_size = off_y-abs((shape[1]-crop[3]))
        crop[3] = shape[1]

    if crop[5]>shape[2]:
        padd_z_size = off_z-abs((shape[2]-crop[5]))
        crop[5] = shape[2]


    sample = np.zeros((off_x,off_y, off_z))
    conditions = [[padd_x>=0, padd_x<off_x],
                  [padd_y>=0, padd_y<off_y],
                  [padd_z>=0, padd_z<off_z],
                  [padd_x_size>0, padd_x_size<=off_x],
                  [padd_y_size>0, padd_y_size<=off_y],
                  [padd_z_size>0, padd_z_size<=off_z]]

    if np.array(conditions).all() == True:
        sample[padd_x:padd_x_size, padd_y:padd_y_size, padd_z:padd_z_size] = np.array(data[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]])[:,:,:,0]

    return resize(sample, ratio=scale_ratio)

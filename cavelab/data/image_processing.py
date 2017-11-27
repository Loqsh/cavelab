from scipy.misc import imresize
from scipy import ndimage
import numpy as np
from math import floor

def resize(image, ratio=(1/3.0, 1/3.0), order=0):
    if ratio[0]==1 and ratio[1]==1:
        return image
    return ndimage.interpolation.zoom(image, ratio, order=order)

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

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


'''
Written by nasimraham
'''
# Elastic transform
def elastic_transformations(alpha, sigma, rng=np.random.RandomState(42),
                            interpolation_order=1):
    """Returns a function to elastically transform multiple images."""
    # Good values for:
    #   alpha: 2000
    #   sigma: between 40 and 60
    def _elastic_transform_2D(images):
        """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
        # Take measurements
        image_shape = images[0].shape
        # Make random fields
        dx = rng.uniform(-1, 1, image_shape) * alpha
        dy = rng.uniform(-1, 1, image_shape) * alpha
        # Smooth dx and dy
        sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
        sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
        # Make meshgrid
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        # Distort meshgrid indices
        distorted_indices = (y + sdy).reshape(-1, 1), \
                            (x + sdx).reshape(-1, 1)

        # Map cooordinates from image to distorted index set
        transformed_images = [map_coordinates(image, distorted_indices, mode='reflect',
                                              order=interpolation_order).reshape(image_shape)
                              for image in images]
        return transformed_images
    return _elastic_transform_2D

'''
    Make black the crop area
'''

def black_crop(image, crop):
    s = image.shape
    image[0:crop,:] = 0
    image[:,0:crop] = 0
    image[s[0]-crop:, :] = 0
    image[:, s[1]-crop:] = 0
    return image

'''
    Get Grid function
    Input shape [x,y]
    Output set of points within batches
'''

def get_grid(shape, step=0, batch_size=8):
        grid = []
        x, y = (0,0)
        step = step

        while(True):
            row = []
            for i in range(batch_size):

                if x>shape[0]:
                    x, y = 0, y+step

                if y>shape[1]:
                    break

                row.append([x,y])
                x += step

            if(len(row)==0):
                break

            while(len(row)<batch_size):
                row.append([-1,-1])

            grid.append(row)
            if row[-1][0] == -1:
                break

        return grid


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


def read_without_borders_3d(data, (x,y,z), (off_x,off_y,off_z), scale_ratio=(1,1,1), voxel_offset = (0,0)):
    shape = np.array(data.shape)
    shape[0] += voxel_offset[0]
    shape[1] += voxel_offset[1]

    crop = [x,x+off_x, y, y+off_y, z, z+off_z]
    (padd_x, padd_x_size, padd_y, padd_y_size, padd_z, padd_z_size) = (0,off_x,0,off_y,0,off_z)

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

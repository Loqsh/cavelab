import h5py
import numpy as np
import image_processing as ip

import csv

class h5data(object):
    def __init__(self, path = ""):
        self.path = path

    '''
        Read 2d data from .h5 without borders, fill black on the borders
    '''
    def read(self, name, (x,y), size, resize=(1/3.0, 1/3.0)):
        with h5py.File(self.path+name+'.h5', 'r') as hf:
            data = hf.get('img')
            shape = hf.get('img').shape
            crop = [x,x+size, y, y+size]
            (padd_x, padd_x_size, padd_y, padd_y_size) = (0,size,0,size)

            if crop[0]<0:
                crop[0]=0
                padd_x = abs(x)

            if crop[2]<0:
                crop[2]=0
                padd_y = abs(y)

            if crop[1]>shape[0]:
                padd_x_size = size-abs((shape[0]-crop[1]))
                crop[1] = shape[0]

            if crop[3]>shape[1]:
                padd_y_size = size-abs((shape[1]-crop[3]))
                crop[3] = shape[1]

            sample = np.zeros((size,size))

            if padd_x>=0 and padd_x<size and padd_y>=0 and padd_y<size and padd_x_size>0 and padd_x_size<=size and padd_y_size>0 and padd_y_size<=size:
                sample[padd_x:padd_x_size, padd_y:padd_y_size] = np.array(data[crop[0]:crop[1], crop[2]:crop[3]])[:,:]

        return ip.resize(sample, resize)

    def get_metadata(self, name = 'registry.txt'):
        f = open(self.path + name, 'rt')
        metadata = list(csv.reader(f, delimiter='\t'))
        return np.array(metadata)

    def get_shape(self, name):
        hf = h5py.File(self.path+name+'.h5', 'r')
        return hf.get('img').shape

    def create_dataset(self, name, shape):
        hf = h5py.File(self.path+name+'.h5')
        dset = hf.create_dataset("img", shape, dtype='uint8')
        return dset

    def write(self, array, (x, y), dset):
        dset[x:x+array.shape[0], y:y+array.shape[1], :] = array

    def write_additive(self, array, (x, y), dset):
        dset[x:x+array.shape[0], y:y+array.shape[1], :] += array

import h5py
import numpy as np

from . import image_processing as ip

import csv

class h5data(object):
    def __init__(self, path = ""):
        self.path = path

    '''
        Read 2d data from .h5 without borders, fill black on the borders
    '''
    def read(self, name, pos, size, resize=(1/3.0, 1/3.0)):
        with h5py.File(self.path+name+'.h5', 'r') as hf:
            data = hf.get('img')

        return ip.read_without_borders_2d(data, pos, size, scale_ratio=resize)

    def get_metadata(self, name = 'registry.txt'):
        f = open(self.path + name, 'rt')
        metadata = list(csv.reader(f, delimiter='\t'))
        return np.array(metadata)

    def get_shape(self, name):
        hf = h5py.File(self.path+name+'.h5', 'r')
        return hf.get('img').shape

    def create_dataset(self, name, shape, dtype='uint8'):
        hf = h5py.File(self.path+name+'.h5')
        dset = hf.create_dataset("img", shape, dtype=dtype)
        return dset

    def write(self, array, x_y, dset):
        (x, y) = x_y
        dset[x:x+array.shape[0], y:y+array.shape[1], :] = array

    def write_additive(self, array, x_y, dset):
        (x, y) = x_y
        dset[x:x+array.shape[0], y:y+array.shape[1], :] += array

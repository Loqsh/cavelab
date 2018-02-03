#from cloudvolume import CloudVolume
import numpy as np
import os
import hashlib
from . import image_processing as ip

class Cloud(object):
    def __init__(self, path = "", mip = 0, bounded = False, fill_missing=True, cache=False):
        self.path = path
        self.mip = mip
        self.path_hash = int(hashlib.sha1(path).hexdigest(), 16) % (10 ** 8)
        self.path_to_cache = '/.cache/'
        self.sampling = 2**self.mip
        self.vol = None
        self.bounded = bounded
        self.fill_missing = fill_missing
        self.cache = cache

        self.vol = CloudVolume(self.path, mip=self.mip, cache=cache, bounded=self.bounded, fill_missing=self.fill_missing)
        self.shape = self.vol.shape


    def read(self, x_y_z, off_x_off_y_off_z):
        (x, y, z) = x_y_z
        (off_x, off_y, off_z) = off_x_off_y_off_z
        #Simple file caching
        name = str(x)+'_'+str(y)+'_'+str(z)+'_'+str((off_x, off_y, off_z))+'_'+str(self.mip)+'_'+str(self.path_hash)
        file_path = self.path_to_cache+name+'.npy'

        if os.path.isfile(file_path) and self.cache:
            return np.load(file_path)[:,:,:,0]

        if not self.vol:
            self.vol = CloudVolume(self.path,  mip=self.mip, bounded=self.bounded, fill_missing=self.fill_missing)
            self.shape = self.vol.shape

        image = self.vol[x:x+off_x, y:y+off_y, z:z+off_z]

        #if self.cache:
        #    np.save(file_path, image)

        return image[:,:,:,0]

    def read_global(self, x_y_z, off_x_off_y_off_z):
        (off_x, off_y, off_z) = off_x_off_y_off_z
        (x, y, z) = x_y_z
        x /= self.sampling
        y /= self.sampling

        image = self.vol[x:x+off_x, y:y+off_y, z:z+off_z]

        return image[:,:,:,0]

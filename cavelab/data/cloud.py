from cloudvolume import CloudVolume
import numpy as np
import os
import hashlib

class Cloud(object):
    def __init__(self, path = "", mip = 0):
        self.path = path
        self.mip = mip
        self.path_hash = int(hashlib.sha1(path).hexdigest(), 16) % (10 ** 8)
        self.path_to_cache = '/.cache/'
        self.sampling = 2**self.mip
        self.vol = CloudVolume(self.path, mip=self.mip)


    def read(self, (x, y, z), size=100, z_width=1):
        #Simple file caching
        name = str(x)+'_'+str(y)+'_'+str(z)+'_'+str(size)+'_'+str(z_width)+'_'+str(self.mip)+'_'+str(self.path_hash)
        file_path = self.path_to_cache+name+'.npy'
        if os.path.isfile(file_path):
            return np.load(file_path)[:,:,:,0]

        x /= self.sampling
        y /= self.sampling
        image = self.vol[x-size:x+size, y-size:y+size, z:z+z_width]/255.0
        np.save(file_path, image)

        return image[:,:,:,0]

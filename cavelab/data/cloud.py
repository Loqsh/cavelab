from cloudvolume import CloudVolume

class Cloud(object):
    def __init__(self, path = "", mip = 0):
        self.path = path
        self.mip = mip
        self.sampling = 2**self.mip
        self.vol = CloudVolume(self.path, mip=self.mip)

    def read(self, (x, y, z), size=100, z_width=1):
        x /= self.sampling
        y /= self.sampling
        image = self.vol[x-size:x+size, y-size:y+size, z:z+z_width]/255.0
        return image[:,:,:,0]

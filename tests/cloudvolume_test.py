from cloudvolume import CloudVolume
from cavelab import visual
mip = 5

vol = CloudVolume('gs://neuroglancer/pinky40_v11/image', mip=mip)
sampling = 2**mip
x, y, z = (59476/sampling, 11570/sampling, 711)
size = 100
image = vol[x-size:x+size, y-size:y+size, z]/255.0
visual.save(image[:,:,0,0], 'pinky_test')

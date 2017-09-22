from cavelab.tf import Graph
import cavelab.data.image_processing as ip
from cavelab.data import visual

import time
from pathos.multiprocessing import ProcessPool, ThreadPool
import numpy as np

class Infer(object):
    def __init__(self, scale=1, batch_size = 8, width = 224, n_threads=1, model_directory="", name="", offset=1536, dtype=np.uint8):
        #Global Parameters
        self.scale = scale
        self.width = width
        self.pool = ThreadPool(n_threads)
        self.model = Graph(directory=model_directory, name=name)
        self.blend_map = ip.get_blend_map(self.width/2, self.width)
        self.offset = offset
        self.done  = []
        self.dtype = np.uint8
        self.batch_size = batch_size

    def read(self, (volume, (x,y,z),  (off_x, off_y, off_z))):
        image = volume.read((x,y,z),  (off_x, off_y, off_z))
        image = ip.resize(image[:,:,0], ratio=(1.0/self.scale, 1.0/self.scale), order=0)/255.0
        visual.save(image, 'infer_input')

        return image

    # Process by batches
    def process(self, volume, (x,y,z), mset):
        t1 = time.time()

        inputs = self.pool.map(self.read, [(volume, (x+i*self.scale*self.width/2, y, z), (self.scale*self.width, self.scale*self.width, 1)) for i in range(8)])
        inputs = np.array(inputs)

        if inputs.mean()==0.0:
            return mset

        images = self.model.process({"image:0": inputs}, ["pred:0"])
        images = [images[0][i, :,:] for i in range(8)]
        images_new = self.pool.map(self.post_process, images)

        i = 0
        for image in images_new:
            x_temp = x+i*self.scale*self.width/2
            mset[x_temp-self.offset:x_temp-self.offset+image.shape[0], y-self.offset:y-self.offset+image.shape[1]] += image[:,:]
            i += 1

        return mset

    def post_process(self, image):
        #image = ip.normalize(image)
        image = image*255
        image = np.multiply(image, self.blend_map).astype(np.uint8)
        image = ip.resize(image, (self.scale,self.scale), order=0)
        return image

    def process_slice(self, input_volume, output_volume, z=0):
        step_x = self.batch_size*self.scale*self.width/2
        step_y = self.scale*self.width/2

        shape_origin = output_volume.shape[0:2]
        shape = output_volume.shape[0:2]+2*self.offset
        mset = np.zeros(shape, np.uint8)

        for x in range(shape[0]//(step_x)-1):
            for y in range(shape[1]//(step_y)-1):
                t1 = time.time()
                mset = self.process(input_volume, (x*step_x-self.offset, y*step_y-self.offset, z), mset)
                t2 = time.time()
                print(str(x)+'/'+str(shape[0]//(step_x)-1),str(y)+'/'+str(shape[1]//(step_y)-1))
        t1 = time.time()
        output_volume.vol[:,:, z] = np.expand_dims(mset[self.offset:self.offset+shape_origin[0],self.offset:self.offset+shape_origin[1]], axis=2)
        t2 = time.time()


    #FIXME improve this to work with cloud volume
    def process_all():
        metadata = get_metadata()
        maximum = 0
        for meta in metadata:
            if meta[0] in done:
                print('pass',meta[0])
                continue
            try:
                if not os.path.isfile(path_s+meta[0]+'.h5'):
                    raise Exception ('Not found')

                if os.path.isfile(path+meta[0]+'.h5'):
                    raise Exception ('Already processed')
                t1 = time.time()
                process_slice(meta[0])
                done.extend(meta[0])
                t2 = time.time()
                print(meta[0], t2-t1)
            except Exception as err:
                print('err',meta[0], err)
        print(maximum)

#process_slice('6,22_prealigned')
#vis.save(util.read(('1,4_prealigned', [29000, 30000], 1536)))
#process_all()

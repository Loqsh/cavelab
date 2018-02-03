from __future__ import print_function

from cavelab.tf import Graph
from cavelab.data import image_processing as ip
from cavelab.data import visual

import time
from pathos.multiprocessing import ProcessPool, ThreadPool
import numpy as np

class Infer(object):
    def __init__(self, in_scale=1,
                       out_scale=1,
                       batch_size = 8,
                       width = 224,
                       n_threads=1,
                       model_directory="",
                       name="",
                       offset=1536,
                       voxel_offset=(0,0),
                       cloud_volume=True,
                       normalization=True,
                       crop_size=20,
                       use_blend_map=False,
                       to_crop=True,
                       features = { "inputs":"", "outputs": ""},
                       chunk_size_z = 64):
        #Global Parameters
        self.offset = offset
        self.voxel_offset = voxel_offset
        self.chunk_size_z = chunk_size_z
        self.in_scale = in_scale
        self.out_scale = out_scale
        self.width = width
        self.pool = ThreadPool(n_threads)
        self.model = Graph(directory=model_directory, name=name)
        self.blend_map = ip.get_blend_map(self.width/2, self.width) #FIXME
        self.batch_size = batch_size
        self.cloud_volume = cloud_volume
        self.normalization = normalization
        self.crop_size = crop_size
        self.use_blend_map = use_blend_map
        self.to_crop = to_crop
        self.features = features

    def read(self, args):
        (volume,  (x,y,z),  (off_x, off_y, off_z), i) = args
        image = ip.read_without_borders_3d(volume.vol,
                                          (x+self.voxel_offset[0],y+self.voxel_offset[1],z),
                                          (off_x, off_y, off_z),
                                          scale_ratio=(1.0/self.scale, 1.0/self.scale, 1),
                                          voxel_offset=self.voxel_offset)
        image = image/255.0
        image = image[:,:,0]

        return image

    # Process by batches
    def process_old(self, volume, x_y_z, mset):
        (x,y,z) = x_y_z
        t1 = time.time()

        inputs = self.pool.map(self.read, [(volume, (x+i*self.scale*self.width/2, y, 0), (self.scale*self.width, self.scale*self.width, 1), i) for i in range(8)])
        inputs = np.array(inputs)
        t2 = time.time()
        print(t2-t1, 'downloading')
        images = self.model.process({"input/image:0": inputs}, ["Passes/image_transformed:0"])
        t3 = time.time()
        images = images[0]
        print(t3-t2, 'processing')
        images = [images[i,:,:,0] for i in range(8)]

        images_new = self.pool.map(self.post_process, images) #

        i = 0
        for image in images_new:
            x_temp = x+i*self.scale*self.width/2
            mset[x_temp+self.offset:x_temp+self.offset+image.shape[0], y+self.offset:y+self.offset+image.shape[1]] += image[:,:]
            i += 1
        t4 = time.time()

        print(t4-t3, 'saving')
        return mset

    '''
    def post_process(self, image):
        if self.normalization:
            image = ip.normalize(image)
        image = np.multiply(image, self.blend_map)
        image = (ip.resize(image, (self.scale, self.scale), order=0)).astype(np.uint8)
        return image
    '''
    def store(self, output_volume, mset, shape_origin, z):
        t1 = time.time()

        if self.cloud_volume:
            output_volume.vol[:,:, z] = np.expand_dims(mset[self.offset:self.offset+shape_origin[0],self.offset:self.offset+shape_origin[1]], axis=2)
        else:
            output_volume[:,:] = mset[self.offset:self.offset+shape_origin[0],self.offset:self.offset+shape_origin[1]]
        t2 = time.time()
        print('saving slice', t2-t1)

    def process_slice(self, input_volume, output_volume, z=0):
        step_x = self.batch_size*self.scale*self.width/2
        step_y = self.scale*self.width/2

        shape_origin = np.array(input_volume.shape[0:2])
        shape = shape_origin+2*self.offset

        mset = np.zeros(shape, np.uint8)

        for x in range(shape[0]//(step_x)):
            for y in range(shape[1]//(step_y)-1):
                print(x*step_x+ self.offset, x*step_x+step_x+ self.offset, y*step_y+ self.offset, y*step_y+step_y+ self.offset)
                t1 = time.time()
                mset = self.process(input_volume, (x*step_x-self.offset, y*step_y-self.offset, z), mset)
                t2 = time.time()
                print(str(x)+'/'+str(shape[0]//(step_x)-1),str(y)+'/'+str(shape[1]//(step_y)-1), t2-t1)

        self.store(output_volume, mset, shape_origin, z)

    '''Optimized inference'''
    def read2d(self, args):
        (volume,  (x,y), step) = args
        image = ip.read_without_borders_2d(volume,
                                          (x, y),
                                          (step, step),
                                          scale_ratio=(1.0/self.in_scale, 1.0/self.in_scale))
        image = image/255.0
        image = image[:,:]

        return image

    def post_process(self, image):
        if self.normalization:
            image = ip.normalize(image)

        if self.use_blend_map:
            image = np.multiply(image, self.blend_map)

        if self.to_crop:
            iamge = ip.black_crop(image, self.crop_size)

        image = ip.resize(image, (self.in_scale, self.in_scale), order=0)

        image = image.astype(np.uint8)

        return image

    def process_core(self, images):
        images = self.model.process({self.features['inputs']: images}, [self.features['outputs']])
        images = images[0]
        images = np.squeeze(images)
        images = [images[i,:,:] for i in range(8)]
        images = self.pool.map(self.post_process, images)
        return images

    def write(self, mset, images, coords, step):
        i = 0
        crop = self.in_scale*self.crop_size
        ishape = images[0].shape[0] - crop
        for i in range(len(coords)):
            if coords[i][0] == -1:
                break
            begin = (coords[i][0]+crop, coords[i][1]+crop)
            end =(coords[i][0]+ishape, coords[i][1]+ishape)
            mset[begin[0]:end[0], begin[1]:end[1]] += images[i][crop:ishape, crop:ishape]
        return mset

    '''

    Given input returns processed output within the same shape
    This process encounters batchification and overlapped cropping

    '''
    def process(self, input_volume):
        #Input organization
        shape_origin = np.array(input_volume.shape)
        width = self.in_scale*self.width

        shape = shape_origin+width
        mset = np.zeros(shape, np.uint8)

        step = width - 2*self.in_scale*self.crop_size
        grid = ip.get_grid(shape_origin+self.in_scale*self.crop_size, step=step, batch_size=self.batch_size)
        i = 0
        for batch in grid:
            t1 = time.time()
            images = self.pool.map(self.read2d,
                        [(input_volume, np.array(coord)-self.in_scale*self.crop_size, width) for coord in batch])
            images = self.process_core(images)
            mset = self.write(mset, images, batch, step)
            t2 = time.time()
            #print(str(i)+'/'+str(len(grid)), t2-t1)
            i += 1
        mset = mset[self.in_scale*self.crop_size:self.in_scale*self.crop_size+shape_origin[0],
                    self.in_scale*self.crop_size:self.in_scale*self.crop_size+shape_origin[1]]
        return mset

    '''

    Locations = [(begin, end)]
        where begin = [x,y,z] and end=[x',y',z]
    '''

    #FIXME Add threaded download/upload
    #FIXME clean up infer.py
    def process_by_superbatch(self, input_volume, output_volume, locations):
        crop = self.in_scale*self.crop_size
        i = 0
        # Make this multithreaded
        for begin, end in locations:
            print(begin, end)
            t1 = time.time()
            data = input_volume.vol[begin[0]-crop:end[0]+crop,begin[1]-crop:end[1]+crop,begin[-1]][:,:,0,0]
            t2 = time.time()
            print(t2-t1, 'download')
            output = self.process(data)
            t3 = time.time()
            output = ip.resize(output[crop:crop+end[0]-begin[0],
                                      crop:crop+end[1]-begin[1]], (self.out_scale, self.out_scale))
            t4 = time.time()
            print(t3-t2, 'process')
            output_volume.vol[self.out_scale*begin[0]:self.out_scale*end[0],
                              self.out_scale*begin[1]:self.out_scale*end[1],
                              begin[-1]] = np.expand_dims(output, axis=2)
            t5 = time.time()
            print(t5-t4, 'upload')
            # Clear cache after processing all 64
            i+=1
            if i%self.chunk_size_z==0:
                i==0
                #input_volume.vol.flush_cache()


    def reactive_process(self, input_volume, output_volume, locations):
        crop = self.in_scale*self.crop_size
        import rx
        from rx import Observable, Observer
        scheduler = rx.concurrency.NewThreadScheduler()
        def download(args):
            (begin,end) = args
            print((begin,end))
            return (np.zeros((190,96)),begin,end)

            return (input_volume.vol[begin[0]-crop:end[0]+crop,\
                             begin[1]-crop:end[1]+crop,\
                             begin[-1]][:,:,0,0],\
                             begin,end)
        def upload(args):
            args = (output,begin,end)
            return 0
            output_volume.vol[self.out_scale*begin[0]:self.out_scale*end[0],
                              self.out_scale*begin[1]:self.out_scale*end[1],
                              begin[-1]] = np.expand_dims(output[crop:crop+end[0]-begin[0],crop:crop+end[1]-begin[1]], axis=2)

        def process(output):
            #output.map(self.process).map(upload).subscribe(on_next=lambda e: print(e),on_error = lambda e: print(e))
            #print(output[0])
            t1 = time.time()
            def fib(x):
                if x ==0 or x==1:
                    return 1
                return fib(x-1)+fib(x-2)
            k = fib(40)
            print(42)
            t2 = time.time()
            return output

        def down(value):
            def go_make_big(subscriber):
                subscriber.on_next(download(value))
                subscriber.on_completed()
            return rx.Observable.create(go_make_big)

        t1 = time.time()
        pipe = Observable.from_(locations)
        pipe.flat_map(lambda x: down(x).subscribe_on(scheduler))\
            .map(process)\
            .map(upload)\
            .subscribe(
                        on_completed=lambda: print("PROCESS 1 done!"),
                        on_error = lambda e: print(e))
        t2 = time.time()
        print('overall',t2-t1)

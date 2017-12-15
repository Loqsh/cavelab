import cavelab as cl
import numpy as np
import tensorflow as tf
import time

class DataGenerator(object):
    def __init__(self,hparams):
        self.input = cl.Cloud(hparams.cloud_src, mip=hparams.cloud_mip, cache=False)
        self.hparams = hparams

    def get_sequence(self):
        range_vector = np.multiply(self.hparams.range[1],np.random.random(3))
        (x,y,z) = np.floor(range_vector+self.hparams.range[0]).astype(int)
        (off_x, off_y,off_z) = (self.hparams.width,self.hparams.width, self.hparams.n_sequence)
        return self.input.read_global((x,y,z),(off_x, off_y,off_z))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(source, filename, size):

  """Converts a dataset to tfrecords."""
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)

  for i in range(size):
      t1 = time.time()
      image = source.get_sequence()
      t2 = time.time()
      print(t2-t1)
      image_raw = np.asarray(image, dtype=np.uint8).tostring()
      ex = tf.train.Example(features=tf.train.Features(feature={
          'image': _bytes_feature(image_raw)
         }))
      writer.write(ex.SerializeToString())
  writer.close()
  print('Saved ', size)


if __name__ == "__main__":
    hparams = cl.hparams(name="preprocessing")
    d = DataGenerator(hparams)
    convert_to(d, hparams.tfrecord_train_dest, hparams.train_size)
    convert_to(d, hparams.tfrecord_test_dest, hparams.test_size)

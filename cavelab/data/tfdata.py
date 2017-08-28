import numpy as np
from cavelab.tf import global_session
import tensorflow as tf
import tf_image_processing as tip


# FIXME: Currently works on pair of search and template images
#        but in ideal case this module should be generalizable
#        Use inheritance per projects or vary feature set

class tfdata(object):
    def __init__(self,  train_file, # train_bad_20.tfrecords
                        features = {
                                    'search_raw': {'in_width': 612, 'width': 324},
                                    'template_raw': {'in_width': 324, 'width':324}
                                   },
                        batch_size = 8,
                        flipping = False,
                        rotating = False,
                        random_crop = False,
                        max_degree = 0):

        self.flipping = flipping
        self.rotating = rotating
        self.train_file = train_file
        self.batch_size = batch_size
        self.features = features
        self.max_degree = max_degree
        self.random_crop = random_crop
        self.s_train, self.t_train = self.inputs(self.batch_size)

    # Functions below modified from here https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
              'search_raw': tf.FixedLenFeature([], tf.string),
              'template_raw': tf.FixedLenFeature([], tf.string),
          })

        # Convert from a scalar string tensor (whose single string has
        search = tf.decode_raw(features['search_raw'], tf.uint8) # Change to tf.int8
        search.set_shape([self.features['search_raw']['in_width']**2])
        search = tf.reshape(search, [self.features['search_raw']['in_width'], self.features['search_raw']['in_width']])

        template = tf.decode_raw(features['template_raw'],tf.uint8) # Change to tf.int8
        template.set_shape([self.features['template_raw']['in_width']**2])
        template = tf.reshape(template, [self.features['template_raw']['in_width'], self.features['template_raw']['in_width']])

        # Rotation - Random Flip left, right, random, up down
        if self.flipping:
          distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
          search = tip.image_distortions(search, distortions)
          template =  tip.image_distortions(template, distortions)

        # Rotation by degree (rotate only single channel)
        if self.rotating:
          angle = tf.random_uniform([1], -self.max_degree,self.max_degree, dtype=tf.float32)
          search = tip.rotate_image(search, angle)

        # Translation Invariance - Crop 712 - > 512 and 324 -> 224
        if self.random_crop:
            search = tf.random_crop(search,  [self.features['search_raw']['width'], self.features['search_raw']['width']])
            template =  tf.random_crop(template, [self.features['template_raw']['width'], self.features['template_raw']['width']])
        else:
            search = tip.central_crop(search,  [self.features['search_raw']['width'], self.features['search_raw']['width']])
            template = tip.central_crop(template, [self.features['template_raw']['width'], self.features['template_raw']['width']])

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        search = tf.cast(search, tf.float32) / 255
        template = tf.cast(template, tf.float32) / 255
        return search, template

    def inputs(self, batch_size, num_epochs=None):
      """Reads input data num_epochs times.
      Args:
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
           train forever.
      Returns:
        A tuple (images, labels), where:
        Note that an tf.train.QueueRunner is added to the graph, which
        must be run using e.g. tf.train.start_queue_runners().
      """

      filename = self.train_file
      with tf.name_scope('input_provider'):
        filename_queue = tf.train.string_input_producer(
            [filename for x in range(1000)], num_epochs=None)

        # Even when reading in multiple threads, share the filename
        # queue.
        search, template = self.read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        search_images, template_images = tf.train.shuffle_batch(
            [search, template], batch_size=batch_size, num_threads=2,
            capacity=1000 * batch_size,
            allow_smaller_final_batch=True,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)


        return search_images, template_images

    def get_batch(self):
        sess = global_session().get_sess()

        search, template = sess.run([self.s_train, self.t_train])
        return search, template

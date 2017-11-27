import numpy as np
from cavelab.tf import global_session
import tensorflow as tf
import tf_image_processing as tip
import image_processing as ip

# FIXME: Implement augmentation for 3D images

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
                        random_brightness = False,
                        max_degree = 0,
                        random_elastic_transform = False):

        self.flipping = flipping
        self.rotating = rotating
        self.train_file = train_file
        self.batch_size = batch_size
        self.features = features
        self.max_degree = max_degree
        self.random_crop = random_crop
        self.random_brightness = random_brightness
        self.random_elastic_transform = random_elastic_transform
        self.outputs = self.inputs(self.batch_size)
        self.elastic_transform = ip.elastic_transformations(2000, 50)

    # Functions below modified from here https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        _features = {feature: tf.FixedLenFeature([], tf.string) for feature in self.features.keys()}
        features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features=_features)

        outputs = {}
        for feature_name, feature in self.features.iteritems():
          # Convert from a scalar string tensor (whose single string has
          image = tf.decode_raw(features[feature_name], tf.uint8) # Change to tf.int8
          if 'depth' in feature:
            shape = [feature['in_width'], feature['in_width'], feature['depth']]
          else:
            shape = [feature['in_width'], feature['in_width']]

          raw_shape = np.prod(shape)
          image.set_shape([raw_shape])
          image = tf.reshape(image, shape)
          outputs[feature_name] = image

        outputs = {k:tf.expand_dims(v, -1) for k, v in outputs.items()}

        # Rotation - Random Flip left, right, random, up down
        if self.flipping:
          outputs = {k: tf.image.random_flip_up_down(v, seed=0) for k, v in outputs.items()}
          outputs = {k: tf.image.random_flip_left_right(v, seed=1) for k, v in outputs.items()}

        if self.random_brightness:
          max_delta = 0.1
          image_name = self.features.keys()[0]
          outputs[image_name] = tf.image.random_brightness(outputs[image_name], max_delta, seed=0)
          outputs[image_name] = tf.image.random_contrast(outputs[image_name], 0.7, 1, seed=0)

        outputs = {k:tf.squeeze(v) for k, v in outputs.items()}
        # Rotation by degree
        if self.rotating:
          angle = tf.random_uniform([1], -self.max_degree,self.max_degree, dtype=tf.float32)
          outputs = {k: tip.rotate_image(v, angle) for k, v in outputs.items()}

        # Translation Invariance - Crop 712 - > 512 and 324 -> 224
        if self.random_crop:
          outputs = {k: tf.random_crop(v, [self.features[k]['width'], self.features[k]['width']], seed=10) for k, v in outputs.items()}
        else:
          outputs = {k: tip.central_crop(v, [self.features[k]['width'], self.features[k]['width']]) for k, v in outputs.items()}

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        outputs = {k: tf.cast(v, tf.float32) / 255.0 for k, v in outputs.items()}
        return outputs.values()

    def inputs(self, batch_size, num_epochs=None):
      """Reads input data num_epochs times.
      Args:
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
           train forever.
      Returns:
        A tuple (images, lab.els), where:
        Note that an tf.train.QueueRunner is added to the graph, which
        must be run using e.g. tf.train.start_queue_runners().
      """

      filename = self.train_file
      with tf.name_scope('input_provider'):
        files = [filename for x in range(1000)]
        flat_list = [item for sublist in files for item in sublist]
        filename_queue = tf.train.string_input_producer(
          flat_list, num_epochs=None)

        # Even when reading in multiple threads, share the filename
        # queue.
        outputs_tensors = self.read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        output = tf.train.shuffle_batch(
            outputs_tensors, batch_size=batch_size, num_threads=2,
            capacity=1000 * batch_size,
            allow_smaller_final_batch=True,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)


        return output

    def get_batch(self):
        sess = global_session().get_sess()
        outputs = sess.run(self.outputs)

        #Elastic Transformation
        if self.random_elastic_transform:
          temp = np.concatenate(outputs, axis=0)
          temp = np.array(self.elastic_transform(temp))
          outputs= np.split(temp, [outputs[0].shape[0]])

        return outputs

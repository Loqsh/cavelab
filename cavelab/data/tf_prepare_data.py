
"""Converts EM data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import src.helpers as helpers
import src.loss as loss
import hyperparams

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data, num_examples, filename,
                  features = {
                              'image': {'in_width': 512, 'width': 512},
                              'label': {'in_width': 512, 'width': 512}
                             }):
  """Converts a dataset to tfrecords."""

  s_rows = features['image'].in_width
  t_rows = features['label'].in_width

  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)


  search_raw = np.asarray(image*255, dtype=np.bool_).tostring()
  temp_raw = np.asarray(label*255, dtype=np.bool_).tostring()

  ex = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(search_raw),
      'label': _bytes_feature(temp_raw),}))

  writer.write(ex.SerializeToString())

  writer.close()

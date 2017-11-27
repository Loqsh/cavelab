import tensorflow as tf
import caffe

sess = tf.Session()
new_saver = tf.train.import_meta_graph("/path/to/checkpoint.meta")
what = new_saver.restore(sess, "/path/to/checkpoint")

all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

conv1 = all_vars[0]
bias1 = all_vars[1]

conv_w1, bias_1 = sess.run([conv1,bias1])

net = caffe.Net('path/to/conv.prototxt', caffe.TEST)

net.params['conv_1'][0].data[...] = conv_w1
net.params['conv_1'][1].data[...] = bias_1

net.save('modelfromtf.caffemodel')

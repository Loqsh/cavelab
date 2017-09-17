from cavelab import tfdata
import tensorflow as tf
from cavelab.tf import global_session, Graph
from cavelab.tf import base_layers as bl
from cavelab.data import image_processing as ip
import numpy as np
from cavelab.data import visual

training_steps = 10000
train_file = '/FilterFinder/data/prepared/bad_trainset_24000_612_324.tfrecords'


def create_model():
    g = Graph()
    # Hparams
    kernel_shape = [[7,7,1,3],
                    [5,5,3,12],
                    [3,3,12,32],
                    [3,3,32,64]]
    batch_size = 8
    size = 128
    learning_rate = 0.0001

    # Input
    g.image = tf.placeholder(tf.float32, shape=[batch_size, size, size], name='image')
    g.template = tf.placeholder(tf.float32, shape=[batch_size, size, size], name='template')
    g.label = tf.placeholder(tf.float32, shape=[batch_size, 2], name='translation')

    # Model here
    g.transformed_template, g.transformed_image = bl.residual_block(g.image, g.template, g.kernels, g.bias, kernel_shape[0])

    conv_1 = bl.cnn(g.transformed_template+g.transformed_image, g.kernels, g.bias, kernel_shape[1])
    pool_1 = bl.max_pool_2x2(conv_1) #64x64

    conv_2 = bl.cnn(pool_1, g.kernels, g.bias, kernel_shape[2])
    pool_2 = bl.max_pool_2x2(conv_2) #32x32

    conv_3 = bl.cnn(pool_2, g.kernels, g.bias, kernel_shape[3])
    pool_3 = bl.max_pool_2x2(conv_3) #16x16

    #FIXME create a new linearizable layer
    W_fc1 = bl.weight_variable([16*16*64, 2])
    b_fc1 = bl.bias_variable([2])

    pool_3_flat = tf.reshape(pool_3, [-1, 16*16*64])
    output = tf.nn.relu(tf.matmul(pool_3_flat, W_fc1) + b_fc1)

    # Loss here
    g.loss = tf.sqrt(tf.reduce_sum(tf.square(output - g.label)))

    # Optimizer
    g.train_step = tf.train.AdamOptimizer(learning_rate).minimize(g.loss)

    return g

def get_training_data(data):
    size = [8, 128, 128]
    x, _ = data.get_batch() # 156x156
    y, offset = ip.random_crop(x, size)
    x = ip.central_crop(x, size)
    label = np.array(offset)[1:3] - (np.array
    ([156, 156]) -np.array(size)[1:3])/2
    label = [label for i in range(8)]
    return x, y, np.array(label)/(156.0-128.0)

def train():
    features = {
                'search_raw': {'in_width': 612, 'width': 156},
                'template_raw': {'in_width': 324, 'width':156}
                }

    data = tfdata(train_file, features=features)
    model = create_model()
    sess = global_session().get_sess()

    try:
        for i in range(training_steps):
            x, y, label = get_training_data(data)
            cl.data.visual.save(x[0], 'tests/a')
            model_run = [model.train_step, model.loss]
            feed_dict = {
                            model.image: x,
                            model.template: y,
                            model.label: label
                        }
            step = sess.run(model_run, feed_dict=feed_dict)
            print(step[1])

    finally:
        global_session().close_sess()

train()

import tensorflow as tf
from math import sqrt

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def kernel_summary(var, name='conv'):
    #tf.get_variable_scope().reuse_variables()
    grid = put_kernels_on_grid (var)
    tf.summary.image(name, grid)

def image_summary(var, name='image'):
    ''' var expected to be [batch_size, width, height]'''

    images = tf.expand_dims(var, dim=3)
    images = tf.image.resize_images(images, [128,128])
    images = tf.transpose(images, [1,2,3,0])
    grid = put_kernels_on_grid(images)
    tf.summary.image(name, grid)

# source from  https://gist.github.com/kukuruza/03731dc494603ceab0c5
def put_kernels_on_grid (kernel, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    shape = tuple(kernel.get_shape().as_list())
    kernel = tf.reshape(kernel, [shape[0], shape[1], shape[2]*shape[3]])
    kernel = tf.expand_dims(kernel, dim=2)

    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: pass #print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1,  tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    axis = -1
    x3 = tf.reshape(x2, tf.concat([tf.expand_dims(t, axis) for t in [grid_X, Y * grid_Y, X, channels]], axis))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    axis = -1
    x5 = tf.reshape(x4, tf.concat([tf.expand_dims(t, axis) for t in [1, X * grid_X, Y * grid_Y, channels]], axis))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (2, 0, 1, 3))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)

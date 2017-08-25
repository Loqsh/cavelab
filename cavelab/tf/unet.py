import tensorflow as tf
import sys
import helpers
import loss
import metrics
from datetime import datetime

def FusionNet(g, hparams):
    with tf.variable_scope('Passes'):
        # Init Convolution Weights
        g.source_alpha = [g.image]
        g.template_alpha = [g.template]

        g.kernel_conv = []
        g.bias = []
        count = hparams.kernel_shape.shape[0]

        # Encode
        print('FusionNet')
        for i in range(count):
            x, y = g.source_alpha[-1], g.template_alpha[-1]
            x, y = helpers.residual_block(x, y, g.kernel_conv, g.bias, hparams.kernel_shape[i])
            g.source_alpha.append(x), g.template_alpha.append(y)

            if i!=(count-1):
                #print(i)
                max_x, max_y = helpers.max_pool_2x2(x), helpers.max_pool_2x2(y)
                g.source_alpha.append(max_x), g.template_alpha.append(max_y)

        # Decode
        for i in range(1, count):
            shape = hparams.kernel_shape[-i]

            shape = [2,2, shape[3], shape[3]/2]
            x, y = g.source_alpha[-1], g.template_alpha[-1]

            x, y = helpers.deconv_block(x, y, g.kernel_conv, g.bias, shape)
            x_enc, y_enc = g.source_alpha[(2*count-3)-2*(i-1)], g.template_alpha[(2*count-3)-2*(i-1)]

            x, y = helpers.residual_block(x+x_enc, y+y_enc, g.kernel_conv, g.bias, hparams.kernel_shape[-i-1])
            g.source_alpha.append(x), g.template_alpha.append(y)

        # Final Layer
        x, y = g.source_alpha[-1], g.template_alpha[-1]
        x, y = helpers.conv_block(x, y, g.kernel_conv, g.bias, [1,1, hparams.kernel_shape[0, 3], hparams.output_layer])
        g.source_alpha.append(tf.identity(x, name="image_transformed")), g.template_alpha.append(tf.identity(y, name="template_transformed"))

        slice_source = tf.squeeze(tf.slice(g.source_alpha[-1], [0, 0, 0, 0], [-1, -1, -1, 1]))
        slice_template = tf.squeeze(tf.slice(g.template_alpha[-1], [0, 0, 0, 0], [-1, -1, -1, 1]))

        if hparams.output_layer > 1:
            #Cross similarity
            g.cross_similarity = helpers.cross_similarity(g.template_alpha[-1])

            slice_source_layers = tf.squeeze(tf.slice(g.source_alpha[-1], [0, 0, 0, 0], [1, -1, -1, -1]))
            slice_source_layers = tf.transpose(slice_source_layers, [2,0,1])
            metrics.image_summary(slice_source_layers, 'search_space_features')

        metrics.image_summary(slice_source, 'search_space')
        metrics.image_summary(slice_template, 'template')

    return g

def Unet(g, hparams):
    # Init Convolution Weights
    g.source_alpha = [g.image]
    g.template_alpha = [g.template]

    # Multilayer convnet
    g.kernel_conv = []
    g.kernel_conv_right = []
    g.kernel_conv_up = []

    g.bias = []
    g.bias_right = []
    g.bias_up = []

    n = hparams.kernel_shape.shape[0]

    # Setup Weights
    with tf.variable_scope('Filters'):
        # Unet weights
        for i in range(n):

            #First layer
            shape = hparams.kernel_shape[i]
            g.kernel_conv, g.bias = helpers.add_conv_weight_layer(g.kernel_conv, g.bias, shape) # Left
            if not i==(n-1): # Right
                shape[2] = 2*shape[3]
                g.kernel_conv_right, g.bias_right = helpers.add_conv_weight_layer(g.kernel_conv_right, g.bias_right, shape )

            #Up layer
            if not i==(n-1):
                new_shape = [2,2, shape[3], 2*shape[3]]
                g.kernel_conv_up, g.bias_up = helpers.add_conv_weight_layer(g.kernel_conv_up, g.bias_up, new_shape) # Up

            #Second layer
            shape[2] = shape[3]
            g.kernel_conv, g.bias = helpers.add_conv_weight_layer(g.kernel_conv, g.bias, shape) # Left
            if not i==(n-1): g.kernel_conv_right, g.bias_right = helpers.add_conv_weight_layer(g.kernel_conv_right, g.bias_right, shape) # Right

    count = len(g.kernel_conv)

    with tf.variable_scope('Passes'):
        # Down
        for i in range(count):
            g.source_alpha.append(helpers.convolve2d(g.source_alpha[-1], g.kernel_conv[i]))
            if not hparams.linear: g.source_alpha[-1] = tf.tanh(g.source_alpha[-1]+g.bias[i])

            g.template_alpha.append(helpers.convolve2d(g.template_alpha[-1], g.kernel_conv[i]))
            if not hparams.linear: g.template_alpha[-1] =tf.tanh(g.template_alpha[-1]+g.bias[i])

            # Max_pooling
            if i%2==0 and i!=count and i!=0: #or i == 2:
                g.source_alpha[-1] = helpers.max_pool_2x2(g.source_alpha[-1])
                g.template_alpha[-1] = helpers.max_pool_2x2(g.template_alpha[-1])

        print('left done')

        # Up
        for i in range(len(g.kernel_conv_up)):
            #Deconvolution
            g.source_alpha.append(helpers.deconv2d(g.source_alpha[-1], g.kernel_conv_up[-i-1]))
            g.template_alpha.append(helpers.deconv2d(g.template_alpha[-1], g.kernel_conv_up[-i-1]))

            #Concatination
            g.source_alpha[-1] = helpers.concat(g.source_alpha[count-2*(i+1)], g.source_alpha[-1])
            g.template_alpha[-1] = helpers.concat(g.template_alpha[count-2*(i+1)], g.template_alpha[-1])

            print('convolutions')
            print(g.source_alpha[-1].get_shape())
            #2 Convolutions
            for j in range(2):
                g.source_alpha.append(helpers.convolve2d(g.source_alpha[-1], g.kernel_conv_right[-2*i-2+j]))
                if not hparams.linear: g.source_alpha[-1] = tf.tanh(g.source_alpha[-1]+g.bias_right[-2*i-2+j])

                g.template_alpha.append(helpers.convolve2d(g.template_alpha[-1], g.kernel_conv_right[-2*i-2+j]))
                if not hparams.linear: g.template_alpha[-1] = tf.tanh(g.template_alpha[-1]+g.bias_right[-2*i-2+j])

        #Output convolution
        g.kernel_conv, g.bias = helpers.add_conv_weight_layer(g.kernel_conv,  g.bias, [1,1,hparams.kernel_shape[0,3],1])

        g.source_alpha.append(helpers.convolve2d(g.source_alpha[-1], g.kernel_conv[-1]))
        g.template_alpha.append(helpers.convolve2d(g.template_alpha[-1], g.kernel_conv[-1]))

        slice_source = tf.squeeze(tf.slice(g.source_alpha[-1], [0, 0, 0, 0], [-1, -1, -1, 1]))
        slice_template = tf.squeeze(tf.slice(g.template_alpha[-1], [0, 0, 0, 0], [-1, -1, -1, 1]))

        metrics.image_summary(slice_source, 'search_space')
        metrics.image_summary(slice_template, 'template')
    return g

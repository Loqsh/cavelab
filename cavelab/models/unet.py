import tensorflow as tf
import sys

import cavelab.tf.base_layers as bl
from datetime import datetime
import numpy as np

def FusionNet(net, kernel_shape = [[3,3,1,8],
                                  [3,3,8,16],
                                  [3,3,16,32],
                                  [3,3,32,64]]):
    #Variable settings
    layers = [net]
    count = len(kernel_shape)

    #Encode
    for i in range(count):

        net = bl.residual_block(net, kernel_shape[i])
        layers.append(net)
        if i!=(count-1):
            net = bl.max_pool_2x2(net)
    # Decode
    for i in range(1, count):
        shape = kernel_shape[-i]
        shape = [2,2, shape[3], shape[3]/2]

        net = bl.deconv_block(net, shape)
        net_enc  = layers[(count-1)-(i-1)]

        shape = kernel_shape[-i-1]
        shape = [3,3, shape[3], shape[3]]
        net = bl.residual_block(net+net_enc, shape)
    return net


def SiameseFusionNet(x, y,  resize=False,
                            kernel_shape = [[3,3,1,8],
                                           [3,3,8,16],
                                           [3,3,16,32],
                                           [3,3,32,64]]):
    #Variable settings
    layers = [(x, y)]
    count = len(kernel_shape)

    #Encode
    for i in range(count):

        x, y = bl.residual_block_dual(x, y, kernel_shape[i])
        layers.append((x, y))
        if i!=(count-1):
            x, y = bl.max_pool_2x2(x), bl.max_pool_2x2(y)
    # Decode
    for i in range(1, count):
        shape = kernel_shape[-i]
        shape = [2,2, shape[3], shape[3]/2]

        x, y = bl.deconv_block_dual(x, y, shape, resize=resize)
        (x_enc, y_enc)  = layers[(count-1)-(i-1)]

        shape = kernel_shape[-i-1]
        shape = [3,3, shape[3], shape[3]]
        x, y = bl.residual_block_dual(x+x_enc, y+y_enc, shape)
    return x, y

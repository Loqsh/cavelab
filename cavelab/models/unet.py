import tensorflow as tf
import sys

import cavelab as cl
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

        net = cl.tf.layers.residual_block(net, kernel_shape[i])
        layers.append(net)
        if i!=(count-1):
            net = cl.tf.layers.max_pool_2x2(net)
    # Decode
    for i in range(1, count):
        shape = kernel_shape[-i]
        shape = [2,2, shape[3], shape[3]/2]

        net = cl.tf.layers.deconv_block(net, shape)
        net_enc  = layers[(count-1)-(i-1)]

        shape = kernel_shape[-i-1]
        shape = [3,3, shape[3], shape[3]]
        net = cl.tf.layers.residual_block(net+net_enc, shape)
    return net

def FusionNet2_5D(net, kernel_shape, activation=tf.tanh):
    #Variable settings
    layers = [net]
    count = len(kernel_shape)

    #Encode
    for i in range(count):
        net = cl.tf.layers.residual_block(net, kernel_shape[i], activation=activation, is3D=True)
        layers.append(net)
        if i!=(count-1):
            net = cl.tf.layers.max_pool_2x2x1(net)
    # Decode
    for i in range(1, count):
        shape = kernel_shape[-i]
        shape = [2,2,1, shape[4], shape[4]/2]
        print(i)
        net = cl.tf.layers.resize_block(net, shape, is3D=True, activation=activation)
        net_enc  = layers[(count-1)-(i-1)]

        shape = kernel_shape[-i-1]
        shape = [3,3,1, shape[4], shape[4]]

        net = cl.tf.layers.residual_block(net+net_enc, shape, activation=activation, is3D=True)

    return net

def SiameseFusionNet(x, y,  resize=False,
                            kernel_shape = [[3,3,1,8],
                                           [3,3,8,16],
                                           [3,3,16,32],
                                           [3,3,32,64]]):
    #Variable settings
    layers = [(x, y)]
    count = len(kernel_shape)
    #xs, ys = [], []
    #Encode
    for i in range(count):

        x, y = cl.tf.layers.residual_block_dual(x, y, kernel_shape[i])
        layers.append((x, y))
        if i!=(count-1):
            x, y = cl.tf.layers.max_pool_2x2(x), cl.tf.layers.max_pool_2x2(y)
    # Decode
    for i in range(1, count):
        shape = kernel_shape[-i]
        shape = [2,2, shape[3], shape[3]/2]

        x, y = cl.tf.layers.deconv_block_dual(x, y, shape, resize=resize)
        (x_enc, y_enc)  = layers[(count-1)-(i-1)]

        shape = kernel_shape[-i-1]
        shape = [3,3, shape[3], shape[3]]
        x, y = cl.tf.layers.residual_block_dual(x+x_enc, y+y_enc, shape)
        xs.insert(0,x), ys.insert(0,y)
    return [xs[-1]], [ys[-1]]

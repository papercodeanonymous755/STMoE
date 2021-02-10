import tensorflow as tf
import numpy as np

import layer_def as ld

def network_gate(inputs, gating_num, moe, activation = 'softmax'):
    ch = inputs.get_shape()[-1]
    H = inputs.get_shape()[1]
    W = inputs.get_shape()[2]
    conv1 = ld.conv_layer(inputs, 4, 2, ch*4, 6, linear = False)
    conv2 = ld.conv_layer(conv1, 4, 2, ch*8, 7, linear = False)
    conv3 = ld.conv_layer(conv2, 4, 2, ch*16, 8, linear = False)
    deconv1 = ld.transpose_conv_layer(conv3, 4, 2, ch*8, 5, linear = False)
    deconv1 = tf.concat([deconv1, conv2], axis = -1)
    deconv2 = ld.transpose_conv_layer(deconv1, 4, 2, 16, 6, linear = False)
    deconv2 = tf.concat([deconv2, conv1], axis= -1)
    deconv3 = ld.transpose_conv_layer(deconv2, 4, 2, 8, 7, linear = False)
    final =  ld.conv_layer(deconv3, 4, 1, gating_num, 10, linear = True)
    final = tf.nn.relu(final)
    sq1 = tf.nn.avg_pool(final, ksize=[1, H, W, 1], strides=[1,1,1,1], padding="VALID")
    sq2 = tf.layers.dense(sq1, units= gating_num // 2,activation = tf.nn.relu)
    if moe == 'moe0':
        sq3 =  tf.layers.dense(sq2, units= gating_num)
        sq3 = tf.nn.softmax(sq3, axis = -1)
        weight  = sq3[:, 0, 0, :]
    else:
        sq3 =  tf.layers.dense(sq2, units= gating_num ,activation = tf.nn.sigmoid)
        excitation = tf.reshape(sq3, [-1,1,1,gating_num])
        weight = final * excitation
        if activation == 'softmax':
            weight = tf.nn.softmax(weight, axis = -1)
    return weight

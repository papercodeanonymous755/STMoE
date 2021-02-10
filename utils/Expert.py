import layer_def as ld
import tensorflow as tf
from SpatioTemporalLSTMCellv2 import SpatioTemporalLSTMCell as stlstm
from MIMBlock import MIMBlock as mimblock
from MIMN import MIMN as mimn
import math
from w_initializer import w_initializer

# shift
def network_shift(inputs):

    conv1 = ld.conv_layer(inputs, 4, 2, 16, 1, linear = False)

    conv2 = ld.conv_layer(conv1, 4, 2, 32, 2, linear = False)

    conv3 = ld.conv_layer(conv2, 4, 2, 64, 3, linear = False)

    conv4 = ld.conv_layer(conv3, 4, 2, 128, 4, linear = False)

    deconv1 = ld.transpose_conv_layer(conv4, 4, 2, 64, 1, linear = False)
    deconv1 = tf.concat([deconv1, conv3], axis = -1)

    deconv2 = ld.transpose_conv_layer(deconv1, 4, 2, 32, 2, linear = False)
    deconv2 = tf.concat([deconv2, conv2], axis = -1)

    deconv3 = ld.transpose_conv_layer(deconv2, 4, 2, 16, 3, linear = False)
    deconv3 = tf.concat([deconv3, conv1], axis = -1)

    deconv4 = ld.transpose_conv_layer(deconv3, 4, 2, 8, 4, linear = False)

    flow = ld.conv_layer(deconv4, 4, 1, 2, 5, linear = True)

    return flow

# rotation
def network_rot(inputs):

    conv1 = ld.conv_layer(inputs, 4, 2, 16, 1, linear = False)

    conv2 = ld.conv_layer(conv1, 4, 2, 32, 2, linear = False)

    conv3 = ld.conv_layer(conv2, 4, 2, 64, 3, linear = False)

    conv4 = ld.conv_layer(conv3, 4, 2, 128, 4, linear = False)

    deconv1 = ld.transpose_conv_layer(conv4, 4, 2, 64, 1, linear = False)
    deconv1 = tf.concat([deconv1, conv3], axis = -1)
    
    deconv2 = ld.transpose_conv_layer(deconv1, 4, 2, 32, 2, linear = False)
    deconv2 = tf.concat([deconv2, conv2], axis = -1)

    deconv3 = ld.transpose_conv_layer(deconv2, 4, 2, 16, 3, linear = False)
    deconv3 = tf.concat([deconv3, conv1], axis = -1)

    deconv4 = ld.transpose_conv_layer(deconv3, 4, 2, 8, 4, linear = False)

    flow = ld.conv_layer(deconv4, 4, 1, 2, 5, linear = True)

    return flow
 
def MIM(inputs, time_step, hidden_state, hidden_state_diff, cell_state, cell_state_diff, st_memory, num_layers, num_hidden, filter_size, last_filter_size, stride = 1, tln = True, trainable_last = False):

    stlstm_layer = []
    stlstm_layer_diff = []
    
    #shape = [B, seq_length, H, W, 1]
    shape = inputs.get_shape().as_list()
    output_channels = 1
    
    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers - 1]

        else:
            num_hidden_in = num_hidden[i - 1]

        if i == 0:
            new_stlstm_layer = stlstm('stlstm'  + str(i + 1), filter_size, num_hidden_in, num_hidden[i], shape, tln = tln)
        else:
            new_stlstm_layer = mimblock('stlstm' + str(i + 1), filter_size, num_hidden_in, num_hidden[i], shape, tln = tln)

        stlstm_layer.append(new_stlstm_layer)
        if time_step == 0:
            cell_state.append(None)
            hidden_state.append(None)

    for i in range(num_layers - 1):
        new_stlstm_layer = mimn('stlstm_diff' + str(i +1), filter_size, num_hidden[i + 1], shape, tln = tln)
        stlstm_layer_diff.append(new_stlstm_layer)
        if time_step == 0:
            cell_state_diff.append(None)
            hidden_state_diff.append(None)

    if time_step == 0:
        st_memory = None

    if time_step > 0:
        reuse = True
    else:
        reuse = False

    with tf.variable_scope('pred_rnn', reuse = reuse):
        preh = hidden_state[0]
        hidden_state[0], cell_state[0], st_memory = stlstm_layer[0](inputs, hidden_state[0], cell_state[0], st_memory)

        for i in range(1, num_layers):
            if time_step > 0:
                if i == 1:
                    hidden_state_diff[i - 1], cell_state_diff[i - 1] = stlstm_layer_diff[i - 1](
                    hidden_state[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                else:
                    hidden_state_diff[i - 1], cell_state_diff[i - 1] = stlstm_layer_diff[i - 1](
                    hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
            else:
                stlstm_layer_diff[i - 1](tf.zeros_like(hidden_state[i - 1]), None, None)
            preh = hidden_state[i]
            hidden_state[i], cell_state[i], st_memory = stlstm_layer[i](
            hidden_state[i - 1], hidden_state_diff[i - 1], hidden_state[i], cell_state[i], st_memory)

        x_gen = tf.layers.conv2d(hidden_state[num_layers - 1],
                                 filters=output_channels,
                                 kernel_size=last_filter_size,
                                 strides=1,
                                 padding='same',
                                 kernel_initializer=w_initializer(num_hidden[num_layers - 1], output_channels),
                                 trainable = trainable_last,
                                 name="back_to_pixel")
        x_gen = tf.nn.relu(x_gen)

    return x_gen, hidden_state, hidden_state_diff, cell_state, cell_state_diff, st_memory

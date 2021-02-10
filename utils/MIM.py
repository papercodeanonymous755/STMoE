from SpatioTemporalLSTMCellv2 import SpatioTemporalLSTMCell as stlstm
from MIMBlock import MIMBlock as mimblock
from MIMN import MIMN as mimn
import math
import tensorflow as tf


def w_initializer(dim_in, dim_out):
    random_range = math.sqrt(6.0 / (dim_in + dim_out))
    return tf.random_uniform_initializer(-random_range, random_range)

def MIM(inputs, B, seq_length, H, W, time_step, hidden_state, hidden_state_diff, cell_state, cell_state_diff, st_memory, num_layers, num_hidden, filter_size,last_filter_size, stride = 1, tln = True):

    stlstm_layer = []
    stlstm_layer_diff = []
    
    shape = [B, seq_length, H, W, 1]
    output_channels = shape[-1]

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
                                 name="back_to_pixel")
        x_gen = tf.nn.relu(x_gen)

    return x_gen, hidden_state, hidden_state_diff, cell_state, cell_state_diff, st_memory

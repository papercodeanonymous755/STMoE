import math

import tensorflow as tf
from TensorLayerNorm import tensor_layer_norm

class SpatioTemporalLSTMCell():
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
                 seq_shape, tln=False, initializer=None):
        """Initialize the basic Conv LSTM cell.
        Args:
            layer_name: layer names for different convlstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden: number of units in output tensor.
            forget_bias: float, The bias added to forget gates (see above).
            tln: whether to apply tensor layer normalization
        """
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden
        self.batch = seq_shape[0]
        self.height = seq_shape[-3]
        self.width = seq_shape[-2]
        self.layer_norm = tln
        self._forget_bias = 1.0

        def w_initializer(dim_in, dim_out):
            random_range = math.sqrt(6.0 / (dim_in + dim_out))
            return tf.random_uniform_initializer(-random_range, random_range)
        if initializer is None or initializer == -1:
            self.initializer = w_initializer
        else:
            self.initializer = tf.random_uniform_initializer(-initializer, initializer)

    def init_state(self):
        return tf.zeros([self.batch, self.height, self.width, self.num_hidden],
                        dtype=tf.float32)

    def __call__(self, x, h, c, m):
        if h is None:
            h = self.init_state()
        if c is None:
            c = self.init_state()
        if m is None:
            m = self.init_state()

        with tf.variable_scope(self.layer_name, reuse = tf.AUTO_REUSE):
            t_cc = tf.layers.conv2d(
                h, self.num_hidden*4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(self.num_hidden_in, self.num_hidden*4),
                name='time_state_to_state')
            s_cc = tf.layers.conv2d(
                m, self.num_hidden*4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(self.num_hidden_in, self.num_hidden*4),
                name='spatio_state_to_state')
            x_shape_in = x.get_shape().as_list()[-1]
            x_cc = tf.layers.conv2d(
                x, self.num_hidden*4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(x_shape_in, self.num_hidden*4),
                name='input_to_state')
            if self.layer_norm:
                t_cc = tensor_layer_norm(t_cc, 'time_state_to_state')
                s_cc = tensor_layer_norm(s_cc, 'spatio_state_to_state')
                x_cc = tensor_layer_norm(x_cc, 'input_to_state')

            i_s, g_s, f_s, o_s = tf.split(s_cc, 4, 3)
            i_t, g_t, f_t, o_t = tf.split(t_cc, 4, 3)
            i_x, g_x, f_x, o_x = tf.split(x_cc, 4, 3)

            i = tf.nn.sigmoid(i_x + i_t)
            i_ = tf.nn.sigmoid(i_x + i_s)
            g = tf.nn.tanh(g_x + g_t)
            g_ = tf.nn.tanh(g_x + g_s)
            f = tf.nn.sigmoid(f_x + f_t + self._forget_bias)
            f_ = tf.nn.sigmoid(f_x + f_s + self._forget_bias)
            o = tf.nn.sigmoid(o_x + o_t + o_s)
            new_m = f_ * m + i_ * g_
            new_c = f * c + i * g
            cell = tf.concat([new_c, new_m],3)
            cell = tf.layers.conv2d(cell, self.num_hidden, 1, 1, padding='same',
                                    kernel_initializer=self.initializer(self.num_hidden*2, self.num_hidden),
                                    name='cell_reduce')
            new_h = o * tf.nn.tanh(cell)

            return new_h, new_c, new_m

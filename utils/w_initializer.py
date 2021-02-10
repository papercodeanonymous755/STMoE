import tensorflow as tf
import math

def w_initializer(dim_in, dim_out):
    random_range = math.sqrt(6.0 / (dim_in + dim_out))
    return tf.random_uniform_initializer(-random_range, random_range)

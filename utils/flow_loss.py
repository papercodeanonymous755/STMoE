import tensorflow as tf
from general import lossfunc

def flow_const(f_g, time_smo, smo, mag, robust = False):
    print(f_g.shape)
    # smoothness loss
    ldudx = tf.losses.mean_squared_error(f_g[:, :,  1:, :, 0], f_g[:, :, :-1, :, 0])
    ldudy = tf.losses.mean_squared_error(f_g[:, :,  :, 1:, 0], f_g[:, :, :, :-1, 0])
    ldvdx = tf.losses.mean_squared_error(f_g[:, :,  1:, :, 1], f_g[:, :, :-1, :, 1])
    ldvdy = tf.losses.mean_squared_error(f_g[:, :,  :, 1:, 1], f_g[:, :, :, :-1, 1])
    SMO = ldudx + ldudy + ldvdx + ldvdy
    if robust:
        first = tf.reduce_mean(lossfunc(f_g[:, :,  1:, :, 0] - f_g[:, :, :-1, :, 0], alpha = tf.constant(0.0), scale = tf.constant(2.0)))
        second = tf.reduce_mean(lossfunc(f_g[:, :,  :, 1:, 0] - f_g[:, :, :, :-1, 0], alpha = tf.constant(0.0), scale = tf.constant(2.0)))
        third = tf.reduce_mean(lossfunc(f_g[:, :,  1:, :, 1] - f_g[:, :, :-1, :, 1], alpha = tf.constant(0.0), scale = tf.constant(2.0)))
        fourth = tf.reduce_mean(lossfunc(f_g[:, :,  :, 1:, 1] - f_g[:, :, :, :-1, 1], alpha = tf.constant(0.0), scale = tf.constant(2.0)))
        SMO = first + second + third + fourth
    # time smoothness loss
    ldt = tf.losses.mean_squared_error(f_g[:, 1:], f_g[:, :-1])
    time_SMO = ldt
    if robust:
        first = tf.reduce_mean(lossfunc(f_g[:, 1:] - f_g[:, :-1], alpha = tf.constant(0.0), scale = tf.constant(2.0)))
        time_SMO = first 

    MAG = tf.losses.mean_squared_error(tf.reduce_mean(f_g), 0)
    if robust:
        MAG = tf.reduce_mean(lossfunc(f_g, alpha = tf.constant(0.0), scale = tf.constant(6.0)))
    return time_smo * time_SMO + smo * SMO + mag * MAG

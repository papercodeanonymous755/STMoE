#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (cpu:-1): ')
import sys
sys.path.append('utils/')
from flow_loss import flow_const
from general import lossfunc 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './models/',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 10,
                            """seq_length""")
tf.app.flags.DEFINE_integer('seq_start', 4,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('num_epoch', 1,
                            """max num of epochs""")
tf.app.flags.DEFINE_float('lr', .001,
                            """learning rate""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_integer('height', 128,
                            """height of input data""")
tf.app.flags.DEFINE_integer('width', 128,
                            """width of input data""")
tf.app.flags.DEFINE_string('warping', 'bilinear',
                            """method of warping""")
tf.app.flags.DEFINE_integer('gating_num', 3,
                            """# of experts""")
tf.app.flags.DEFINE_string('activation', 'softmax',
                            """gating activation""")
tf.app.flags.DEFINE_boolean('restore_all', False, 
                            """whether restore all or not """)
tf.app.flags.DEFINE_boolean('restore_gating', False, 
                            """whether restore gating or not """)
tf.app.flags.DEFINE_string('training', 'gating', 
                            """gating or the whole model""")
tf.app.flags.DEFINE_float('mag', 0.005,
                            """loss weight for flow mag""")
tf.app.flags.DEFINE_float('time_smo', 0.005,
                            """loss weight for flow time_smo""")
tf.app.flags.DEFINE_float('smo', 0.005,
                            """loss weight for flow smo""")
tf.app.flags.DEFINE_integer('h_conv_ksize', 1,
                            """kernel size of H""")
tf.app.flags.DEFINE_float('w_ent', 0.01,
                            """loss weight for w_entropy""")
tf.app.flags.DEFINE_boolean('robust', True,
                            """use robust func or not """)
tf.app.flags.DEFINE_float('w_time_smo', 0.01,
                            """loss weight for w_time_smo""")
tf.app.flags.DEFINE_float('robust_x', 0.9,
                            """hyper-parameter of robust func""")
tf.app.flags.DEFINE_boolean('trainable_last', True, 
                            """train backtopixel conv(H) of MIM or not""")
tf.app.flags.DEFINE_string('f', '', 'kernel')

import layer_def as ld
from warp import get_pixel_value
from warp import bilinear_warp as tf_warp
import w_initializer

from Expert import network_shift
from Expert import MIM
network_rot = MIM
network_grow = MIM
from Gating import network_gate

network_gate = tf.make_template('gate', network_gate)
network_shift = tf.make_template('network_convdeconv', network_shift)

def train(model1, model2, model3, model4, model5, train_data, valid_data):
    """
    model1: model path for translation
    model2: model path for rotation
    model3: model path for growth/decay
    model4: model path for gating
    model5: model path for the whole STMoE model
    train_data: np.array for training
    valid_data: np.array for validation
    """
    x= tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.height, FLAGS.width, FLAGS.seq_length])
    x_g = []
    weights = []
    f_g_s = []
    x_g_s, x_g_r, x_g_g = [], [], []
    
    hidden_state_1, hidden_state_diff_1, cell_state_1, cell_state_diff_1, st_memory_1 = [], [], [], [], []
    hidden_state_2, hidden_state_diff_2, cell_state_2, cell_state_diff_2, st_memory_2 = [], [], [], [], []        
    
    for i in range(FLAGS.seq_start - 1):
            with tf.variable_scope('expert2'):
                inputs = x[:, :, :, i]
                inputs = inputs[:, :, :, tf.newaxis]
                x_generate_r, hidden_state_1, hidden_state_diff_1, cell_state_1, cell_state_diff_1, st_memory_1 =                     network_grow(inputs, i, hidden_state_1, hidden_state_diff_1, cell_state_1, cell_state_diff_1, st_memory_1, 3, [8, 8, 8], (3, 3), FLAGS.h_conv_ksize, stride=1, tln = True, trainable_last = FLAGS.trainable_last)
                
            with tf.variable_scope('expert3'):
                inputs = x[:, :, :, i]
                inputs = inputs[:, :, :, tf.newaxis]
                x_generate_g, hidden_state_2, hidden_state_diff_2, cell_state_2, cell_state_diff_2, st_memory_2 =                     network_grow(inputs, i, hidden_state_2, hidden_state_diff_2, cell_state_2, cell_state_diff_2, st_memory_2, 3, [8, 8, 8], (3, 3), FLAGS.h_conv_ksize, stride=1, tln = True, trainable_last = FLAGS.trainable_last)      
    
    # predict recursively
    for i in range(FLAGS.seq_length - FLAGS.seq_start):
        print('frame_{}'.format(i))
        if i == 0:
            with tf.variable_scope('expert1'):
                f_generate = network_shift(x[:, :, :, i:i+FLAGS.seq_start])
                f_g_s.append(f_generate)
                last_x = x[:, :, :, FLAGS.seq_start - 1]
                x_generate_s = tf_warp(last_x[:, :, :, tf.newaxis], f_generate, FLAGS.height, FLAGS.width)
                x_generate_s = tf.reshape(x_generate_s[:, :, :, 0], [FLAGS.batch_size, FLAGS.height, FLAGS.width, 1])
                x_g_s.append(x_generate_s)

            with tf.variable_scope('expert2'):
                inputs = x[:, :, :,  FLAGS.seq_start - 1]
                inputs = inputs[:, :, :, tf.newaxis]
                x_generate_r, hidden_state_1, hidden_state_diff_1, cell_state_1, cell_state_diff_1, st_memory_1 =                      network_grow(inputs, i + FLAGS.seq_start, hidden_state_1, hidden_state_diff_1, cell_state_1, cell_state_diff_1, st_memory_1, 3, [8, 8, 8], (3, 3), FLAGS.h_conv_ksize, stride=1, tln = True, trainable_last = FLAGS.trainable_last)
                x_g_r.append(x_generate_r)

            with tf.variable_scope('expert3'):
                inputs = x[:, :, :,  FLAGS.seq_start - 1]
                inputs = inputs[:, :, :, tf.newaxis]
                x_generate_g, hidden_state_2, hidden_state_diff_2, cell_state_2, cell_state_diff_2, st_memory_2 =                      network_grow(inputs, i + FLAGS.seq_start, hidden_state_2, hidden_state_diff_2, cell_state_2, cell_state_diff_2, st_memory_2, 3, [8, 8, 8], (3, 3), FLAGS.h_conv_ksize, stride=1, tln = True, trainable_last = FLAGS.trainable_last)
                x_g_g.append(x_generate_g)             

            with tf.variable_scope('gating_network'):
                weight = network_gate(x[:, :, :, i:i+FLAGS.seq_start], FLAGS.gating_num, moe='moe1')
                x_sr = tf.concat([x_generate_s, x_generate_r, x_generate_g], axis = -1)
                x_generate = weight * x_sr
                x_generate = tf.reduce_sum(x_generate, axis=-1)
                x_g.append(x_generate)
                weights.append(weight)

        else:
            x_gen = tf.stack(x_g)
            print(x_gen.shape)
            x_gen = tf.transpose(x_gen, [1, 2, 3, 0])

            if i < FLAGS.seq_start:
                x_input = tf.concat([x[:, :, :, i:FLAGS.seq_start], x_gen[:,:, :, :i]], axis = 3)
            else:
                x_input = x_gen[:, :, :, i - FLAGS.seq_start:i]

            with tf.variable_scope('expert1'):
                f_generate = network_shift(x_input)
                f_g_s.append(f_generate)
                last_x = x_g[-1]
                x_generate_s = tf_warp(last_x[:, :, :, tf.newaxis], f_generate,  FLAGS.height, FLAGS.width)
                x_generate_s = tf.reshape(x_generate_s[:, :, :, 0], [FLAGS.batch_size, FLAGS.height, FLAGS.width, 1])
                x_g_s.append(x_generate_s)

            with tf.variable_scope('expert2'):
                inputs = x_g[-1]
                inputs = inputs[:, :, :, tf.newaxis]
                x_generate_r, hidden_state_1, hidden_state_diff_1, cell_state_1, cell_state_diff_1, st_memory_1 =                      network_grow(inputs, i + FLAGS.seq_start, hidden_state_1, hidden_state_diff_1, cell_state_1, cell_state_diff_1, st_memory_1, 3, [8, 8, 8], (3, 3), FLAGS.h_conv_ksize, stride=1, tln = True, trainable_last = FLAGS.trainable_last)
                x_g_r.append(x_generate_r)

            with tf.variable_scope('expert3'):
                inputs = x_g[-1]
                inputs = inputs[:, :, :, tf.newaxis]
                x_generate_g, hidden_state_2, hidden_state_diff_2, cell_state_2, cell_state_diff_2, st_memory_2 =                      network_grow(inputs, i + FLAGS.seq_start, hidden_state_2, hidden_state_diff_2, cell_state_2, cell_state_diff_2, st_memory_2, 3, [8, 8, 8], (3, 3), FLAGS.h_conv_ksize, stride=1, tln = True, trainable_last = FLAGS.trainable_last)
                x_g_g.append(x_generate_g) 

            with tf.variable_scope('gating_network'):
                weight = network_gate(x_input, FLAGS.gating_num, moe='moe1')
                x_sr = tf.concat([x_generate_s, x_generate_r, x_generate_g], axis = -1)
                x_generate = weight*x_sr
                x_generate = tf.reduce_sum(x_generate, axis=-1)
                x_g.append(x_generate) 
                weights.append(weight)
                
    x_g = tf.stack(x_g)
    x_g = tf.transpose(x_g, [1, 2, 3, 0])

    f_g_s = tf.stack(f_g_s)
    f_g_s = tf.transpose(f_g_s, [1, 0, 2, 3, 4])

    weights = tf.stack(weights)
    weights = tf.transpose(weights, [1, 0, 2, 3, 4])
    
    # build a saver 
    expert1_varlist = {v.op.name.lstrip("expert1/"): v 
          for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="expert1/")} 

    expert1_saver = tf.train.Saver(var_list=expert1_varlist) 
    
    expert2_varlist = {v.op.name[8:]: v 
          for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="expert2/")} 
    expert2_saver = tf.train.Saver(var_list=expert2_varlist) 
    
    expert3_varlist = {v.op.name[8:]: v 
          for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="expert3/")} 
    expert3_saver = tf.train.Saver(var_list=expert3_varlist)   
    
    # build a gating saver
    gating_varlist = {v.name.lstrip("gating_network/"): v 
           for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="gating_network/")} 
    gating_saver = tf.train.Saver(var_list=gating_varlist, max_to_keep=100)    

    # w time smoothness loss
    wdt = tf.losses.mean_squared_error(weights[:, 1:], weights[:, :-1])
    
    # MSE loss
    MSE = tf.losses.mean_squared_error(x[:, :, :, FLAGS.seq_start:], x_g[:, :, :, :])
   
    # loss func
    if FLAGS.training == 'all':
        first = tf.reduce_mean(lossfunc(x[:, :, :, FLAGS.seq_start:] - x_g[:, :, :, :], alpha = tf.constant(0.0), scale = tf.constant(FLAGS.robust_x)))
        second = tf.reduce_mean(lossfunc(-tf.log(weights + 1e-10)*weights, alpha = tf.constant(0.0), scale = tf.constant(0.3)))
        third = flow_const(f_g_s, FLAGS.time_smo, FLAGS.smo, FLAGS.mag, robust = True)
        fourth = tf.reduce_mean(lossfunc(weights[:, 1:] - weights[:, :-1], alpha = tf.constant(0.0), scale = tf.constant(0.75)))
        loss = first + FLAGS.w_ent * second + third + FLAGS.w_time_smo * fourth
    else:
        first = tf.reduce_mean(lossfunc(x[:, :, :, FLAGS.seq_start:] - x_g[:, :, :, :], alpha = tf.constant(0.0), scale = tf.constant(FLAGS.robust_x)))
        second = tf.reduce_mean(lossfunc(-tf.log(weights + 1e-10)*weights, alpha = tf.constant(0.0), scale = tf.constant(0.3)))
        fourth = tf.reduce_mean(lossfunc(weights[:, 1:] - weights[:, :-1], alpha = tf.constant(0.0), scale = tf.constant(0.75)))            
        loss = first + FLAGS.w_ent * second + FLAGS.w_time_smo * fourth
    

    if FLAGS.training == 'gating':
        train_var = list(tf.get_collection(tf.GraphKeys.VARIABLES, scope="gating_network/"))
    elif FLAGS.training == 'all':
        train_var = list(tf.get_collection(tf.GraphKeys.VARIABLES))
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss, var_list = train_var)

    # List of all varables
    variables = tf.global_variables()

    # strat rinning operations on Graph
    sess = tf.Session()
    init = tf.global_variables_initializer()
    
    print('init netwrok from scratch....')
    sess.run(init)
                         
    expert1_saver.restore(sess, model1)

    expert2_saver.restore(sess, model2)
    
    expert3_saver.restore(sess, model3)
    
    # restore gating saver
    if FLAGS.restore_gating:
        gating_saver.restore(sess, model4)
    
    # restore all saver
    all_saver =  tf.train.Saver(max_to_keep=100)
    
    if FLAGS.restore_all:
        all_saver.restore(sess, model5)
            
    Loss_MSE, Loss_MSE_v  = [], []
    all_Loss, all_Loss_v = [], []
    
    np.random.seed(2020)

    # train
    for epoch in range(FLAGS.num_epoch):
        loss_MSE, loss_MSE_v = [], []
        loss_all_epoch, loss_all_epoch_v = [], []
        sff_idx = np.random.permutation(train_data.shape[0])
        sff_idx_v = np.random.permutation(valid_data.shape[0])
        # train
        for idx in range(0, train_data.shape[0], FLAGS.batch_size):  
            if idx + FLAGS.batch_size < train_data.shape[0]:
                batch_x = train_data[sff_idx[idx:idx + FLAGS.batch_size]]
                batch_x = batch_x.transpose(0, 2, 3, 1)
                __, train_loss_all, train_mse = sess.run([train_op, loss, MSE], feed_dict = {x:batch_x})
                loss_MSE.append(train_mse)
                loss_all_epoch.append(train_loss_all)
        
        # validation 
        for idx in range(0, valid_data.shape[0], FLAGS.batch_size):   
            if idx + FLAGS.batch_size < valid_data.shape[0]:
                batch_x = valid_data[sff_idx_v[idx:idx + FLAGS.batch_size]]
                batch_x = batch_x.transpose(0, 2, 3, 1)
                valid_all_loss, valid_mse = sess.run([loss, MSE], feed_dict = {x:batch_x})
                loss_MSE_v.append(valid_mse)
                loss_all_epoch_v.append(valid_all_loss)

        Loss_MSE.append(np.mean(loss_MSE))
        Loss_MSE_v.append(np.mean(loss_MSE_v))
        all_Loss.append(np.mean(loss_all_epoch))
        all_Loss_v.append(np.mean(loss_all_epoch_v))       
        print('epoch, MSE, valid_MSE:{} {} {}'.format(epoch, Loss_MSE[-1], Loss_MSE_v[-1]))
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'STMoE-1_{}'.format(FLAGS.training))
            
            if FLAGS.training == 'gating':
                print('save gating saver')
                gating_saver.save(sess, checkpoint_path, global_step = epoch + 1)
                
            elif FLAGS.training == 'all':
                print('save all saver')
                all_saver.save(sess, checkpoint_path, global_step = epoch + 1)


def main():
    model1 = './models/STMoE-1_trans'
    model2 = './models/STMoE-1_rot'
    model3 = './models/STMoE-1_grow' 
    model4 = 'none'
    model5 = 'none'

    da = np.load('./data/DynamicMNIST.npy')
    train_data = da[:2000]
    valid_data = da[2000:3000]
    train_data = train_data[::8]
    max_intensity = 255
        
    train_data = train_data / max_intensity
    valid_data = valid_data / max_intensity
    
    train(model1, model2, model3, model4, model5, train_data, valid_data)


if __name__ == '__main__':
    main()




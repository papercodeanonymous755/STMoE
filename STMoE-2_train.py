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
tf.app.flags.DEFINE_integer('h_conv_ksize', 3,
                            """kernel size of H""")
tf.app.flags.DEFINE_float('w_ent', 0.01,
                            """loss weight for w_entropy""")
tf.app.flags.DEFINE_boolean('robust', True,
                            """use robust func or not """)
tf.app.flags.DEFINE_float('w_time_smo', 0.01,
                            """loss weight for w_time_smo""")
tf.app.flags.DEFINE_float('robust_x', 0.9,
                            """hyper-parameter of robust func""")
tf.app.flags.DEFINE_boolean('trainable_last', False, 
                            """train backtopixel conv(H) of MIM or not""")
tf.app.flags.DEFINE_string('f', '', 'kernel')

import layer_def as ld
from w_initializer import w_initializer
from Expert import MIM
from Gating import network_gate
network_gate = tf.make_template('gate', network_gate)

def train(model1, model2, model3, model4, model5, model6, train_data, valid_data):
    """
    model1: model path for translation
    model2: model path for rotation
    model3: model path for growth/decay
    model4: model path for H (decoder)
    model5: model path for gating
    model6: model path for the whole STMoE model
    train_data: np.array for training
    valid_data: np.array for validation
    """
    x= tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.height, FLAGS.width, FLAGS.seq_length])
    x_g = []
    weights = []
    h_s, h_r, h_g, h_g_all= [], [], [], []

    hidden_state, hidden_state_diff, cell_state, cell_state_diff, st_memory = [], [], [], [], []
    hidden_state_r, hidden_state_diff_r, cell_state_r, cell_state_diff_r, st_memory_r = [], [], [], [], []
    hidden_state_s, hidden_state_diff_s, cell_state_s, cell_state_diff_s, st_memory_s = [], [], [], [], []  
    
    for i in range(FLAGS.seq_start - 1):
            
            with tf.variable_scope('expert2'):
                inputs = x[:, :, :, i]
                inputs = inputs[:, :, :, tf.newaxis]
                x_generate_r, hidden_state_r, hidden_state_diff_r, cell_state_r, cell_state_diff_r, st_memory_r =                     MIM(inputs, i, hidden_state_r, hidden_state_diff_r, cell_state_r, cell_state_diff_r, st_memory_r, 3, [8, 8, 8], (3, 3), last_filter_size = FLAGS.h_conv_ksize, stride=1, tln = True)

            with tf.variable_scope('expert1'):
                inputs = x[:, :, :, i]
                inputs = inputs[:, :, :, tf.newaxis]
                x_generate_s, hidden_state_s, hidden_state_diff_s, cell_state_s, cell_state_diff_s, st_memory_s =                     MIM(inputs, i, hidden_state_s, hidden_state_diff_s, cell_state_s, cell_state_diff_s, st_memory_s, 3, [8, 8, 8], (3, 3), last_filter_size = FLAGS.h_conv_ksize, stride=1, tln = True)
            
            with tf.variable_scope('expert3'):
                inputs = x[:, :, :, i]
                inputs = inputs[:, :, :, tf.newaxis]
                x_generate_g, hidden_state, hidden_state_diff, cell_state, cell_state_diff, st_memory =                     MIM(inputs, i, hidden_state, hidden_state_diff, cell_state, cell_state_diff, st_memory, 3, [8, 8, 8], (3, 3), last_filter_size = FLAGS.h_conv_ksize, stride=1, tln = True)
                
    for i in range(FLAGS.seq_length - FLAGS.seq_start):
        print('frame_{}'.format(i))
        if i == 0:
            inputs = x[:, :, :,  FLAGS.seq_start - 1]
            inputs = inputs[:, :, :, tf.newaxis]
            last_x = x[:, :, :, FLAGS.seq_start - 1]
            last_x = last_x[:, :, :, tf.newaxis]
            x_input = x[:, :, :, i:FLAGS.seq_start]
        else:
            inputs = x_g[-1]
            last_x = x_g[-1]
            x_gen = tf.stack(x_g)
            print(x_gen.shape)
            x_gen = tf.transpose(x_gen[:, :, :, :, 0], [1, 2, 3, 0])

            if i < FLAGS.seq_start:
                x_input = tf.concat([x[:, :, :, i:FLAGS.seq_start], x_gen[:,:, :, :i]], axis = 3)
            else:
                x_input = x_gen[:, :, :, i - FLAGS.seq_start:i]
        with tf.variable_scope('expert3'):
            x_generate_g, hidden_state, hidden_state_difff, cell_state, cell_state_diff, st_memory =                 MIM(inputs, i + FLAGS.seq_start - 1, hidden_state, hidden_state_diff, cell_state, cell_state_diff, st_memory, 3, [8, 8, 8], (3, 3), last_filter_size = FLAGS.h_conv_ksize, stride=1, tln = True)
            h_g.append(hidden_state[-1])
            
        with tf.variable_scope('expert1'):  
            x_generate_s, hidden_state_s, hidden_state_difff_s, cell_state_s, cell_state_diff_s, st_memory_s =                 MIM(inputs, i + FLAGS.seq_start - 1, hidden_state_s, hidden_state_diff_s, cell_state_s, cell_state_diff_s, st_memory_s, 3, [8, 8, 8], (3, 3), last_filter_size = FLAGS.h_conv_ksize, stride=1, tln = True)
            h_s.append(hidden_state_s[-1])
            
        with tf.variable_scope('expert2'):   
            x_generate_r, hidden_state_r, hidden_state_difff_r, cell_state_r, cell_state_diff_r, st_memory_r =                 MIM(inputs, i + FLAGS.seq_start - 1, hidden_state_r, hidden_state_diff_r, cell_state_r, cell_state_diff_r, st_memory_r, 3, [8, 8, 8], (3, 3), last_filter_size = FLAGS.h_conv_ksize, stride=1, tln = True)
            h_r.append(hidden_state_r[-1]) 

        
        with tf.variable_scope('gating_network', reuse = tf.AUTO_REUSE):
            if i == 0:
                x_input = x[:, :, :, i:i+FLAGS.seq_start]
            elif i < FLAGS.seq_start:
                x_input = tf.concat([x[:, :, :, i:FLAGS.seq_start], x_gen[:,:, :, :i]], axis = 3)
            else:
                x_input = x_gen[:, :, :, i - FLAGS.seq_start:i]
            
            weight = network_gate(x_input, FLAGS.gating_num, FLAGS.activation)
            h_generate = weight[:, :, :, 0][:, :, :, tf.newaxis] * h_s[-1] + weight[:, :, :, 1][:, :, :, tf.newaxis] * h_r[-1] + weight[:, :, :, 2][:, :, :, tf.newaxis] * h_g[-1] 
            
            x_generate = tf.layers.conv2d(h_generate,
                                 filters= 1,
                                 kernel_size=FLAGS.h_conv_ksize,
                                 strides=1,
                                 padding='same',
                                 kernel_initializer=w_initializer(8, 1),
                                 trainable = False,
                                 name="back_to_pixel")
            h_g_all.append(h_generate)
            x_g.append(x_generate)
            weights.append(weight)
            x_gen = tf.stack(x_g)
            print('x_gen', x_gen.shape)
            x_gen = tf.transpose(x_gen[:, :, :, :, 0], [1, 2, 3, 0])

    x_g = tf.stack(x_g)
    x_g = tf.transpose(x_g[:, :, :, :, 0], [1, 2, 3, 0])
    print(x_g.shape)
    
    weights = tf.stack(weights)


    MSE = tf.losses.mean_squared_error(x[:, :, :, FLAGS.seq_start:], x_g[:, :, :, :])
    
    # loss func
    first = tf.reduce_mean(lossfunc(x[:, :, :, FLAGS.seq_start:] - x_g[:, :, :, :], alpha = tf.constant(0.0), scale = tf.constant(FLAGS.robust_x)))
    second = tf.reduce_mean(lossfunc(-tf.log(weights + 1e-10)*weights, alpha = tf.constant(0.0), scale = tf.constant(0.3)))
    third = tf.reduce_mean(lossfunc(weights[:, 1:] - weights[:, :-1], alpha = tf.constant(0.0), scale = tf.constant(0.75)))
    loss = first + FLAGS.w_ent * second +  FLAGS.w_time_smo * third

    # training
    if FLAGS.training == 'all':
        train_var = list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        print('optimize all network')
        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss, var_list = train_var)

    elif FLAGS.training == 'gating':
        print('optimize gating network')
        train_var = list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gating_network/"))
        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss, var_list = train_var)
    
    # List of all varables
    variables = tf.global_variables()
    # strat rinning operations on Graph
    sess = tf.Session()
    init = tf.global_variables_initializer()
    print('init netwrok from scratch....')
    sess.run(init)
    

    # restore experts or build train saver
    expert1_varlist = {v.op.name[8:]: v
          for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="expert1/")}
    expert1_saver = tf.train.Saver(var_list=expert1_varlist)

    expert2_varlist = {v.op.name[8:]: v
          for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="expert2/")}
    expert2_saver = tf.train.Saver(var_list=expert2_varlist)

    expert3_varlist = {v.op.name[8:]: v
          for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="expert3/")}
    expert3_saver = tf.train.Saver(var_list=expert3_varlist)

    # restore the saver
    expert1_saver.restore(sess, model1)

    expert2_saver.restore(sess, model2)
    
    expert3_saver.restore(sess, model3)
    
    # build and restore gating network 
    gating_varlist = {v.name.lstrip("gating_network/"): v
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gating_network/")}
    gating_saver = tf.train.Saver(var_list=gating_varlist, max_to_keep=100)
    
    if FLAGS.restore_gating:
        gating_saver.restore(sess, model5)
        
    # restore back to pixel (H)
    back2pixcel_varlist = {'pred_rnn/'+ v.op.name[15:]: v
      for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="gating_network/back_to_pixel")}
    print(back2pixcel_varlist)
    back2pixel_saver = tf.train.Saver(var_list=back2pixcel_varlist)
    back2pixel_saver.restore(sess, model4)
    
    # build all saver
    all_varlist = {v.name: v
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}
    all_saver =  tf.train.Saver(all_varlist, max_to_keep=100)

    if FLAGS.restore_all:
        all_saver.restore(sess, model6)
            
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
            checkpoint_path = os.path.join(FLAGS.train_dir, 'STMoE-2_{}'.format(FLAGS.training))
            
            if FLAGS.training == 'gating':
                print('save gating saver')
                gating_saver.save(sess, checkpoint_path, global_step = epoch + 1)
                
            elif FLAGS.training == 'all':
                print('save all saver')
                all_saver.save(sess, checkpoint_path, global_step = epoch + 1)

def main():
    model1 = './models/STMoE-2_trans'
    model2 = './models/STMoE-2_rot'
    model3 = './models/STMoE-2_grow' 
    model4 ='./models/STMoE-2_H' 
    model5 = 'none'
    model6 = 'none'

    train_data = np.load('./data/train_data.npy')
    valid_data = np.load('./data/valid_data.npy')
    max_intensity = 255
        
    train_data = train_data / max_intensity
    valid_data = valid_data / max_intensity
    
    train(model1, model2, model3, model4, model5, model6, train_data, valid_data)


if __name__ == '__main__':
    main()




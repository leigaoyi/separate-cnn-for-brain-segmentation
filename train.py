# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:50:59 2018

@author: ky
"""

from load_data import create_input_data

from model import build_model
from model import critic
from ops import mask_seg_input

import numpy as np
import tensorflow as tf
import os 
import time

BATCH_SIZE = 10
lr = 0.0001
beta1 = 0.9
epoch = 50
check_dir = './checkpoints/'

input_holder = tf.placeholder(tf.float32, [BATCH_SIZE, 128, 128, 4])
label_holder = tf.placeholder(tf.float32, [BATCH_SIZE, 128, 128, 1])

if not os.path.exists(check_dir):
    os.makedirs(check_dir)

#================build_model============
model_predict = build_model(input_holder)
false_mask = mask_seg_input(input_holder, model_predict)
fake_last_layer = tf.reduce_mean(tf.abs(false_mask[-1]))
true_mask = mask_seg_input(input_holder, label_holder)
critic_true = critic(true_mask)
critic_false = critic(false_mask, reuse=True)
#================dice function============
def dice_coe(output, target, loss_type='jaccard', axis=(0, 1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice
#==================== loss funtion ===============
dice_score = dice_coe(model_predict, label_holder)
seg_loss = 1 - dice_score

adverse_loss = 0
for i in range(len(critic_true)):
    layer_abs = tf.abs(critic_true[i]-critic_false[i])
    layer_mean = tf.reduce_mean(layer_abs)
    adverse_loss += layer_mean

loss = seg_loss + 1 * adverse_loss
#==============optimization function============

variable_list = tf.trainable_variables()
optimization = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(loss, var_list=variable_list)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print('init success ')
#-----------count parameters number-------------
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
#    print(shape)
#    print(len(shape))
    variable_parameters = 1
    for dim in shape:
#        print(dim)
        variable_parameters *= dim.value
#    print(variable_parameters)
    total_parameters += variable_parameters
print('total number of parameters: ',total_parameters)

saver = tf.train.Saver(max_to_keep=1)

#=============training=====================
def main(task):
    step = 0
    data, label = create_input_data() #[?, 180, 180, 4] --> random crop 128 in training
    shuffle = [i for i in range(data.shape[0])]
    np.random.shuffle(shuffle)
    data = [data[i] for i in shuffle]
    label = [label[i] for i in shuffle]
    data = np.asarray(data, dtype=np.float32)
    label = np.asarray(label)
    
    if task == 'all':
        label = (label>0).astype(np.float32)
    if task == 'necrotic':
        label = (label==1).astype(np.float32)
    if task == 'edema':
        label = (label==2).astype(np.float32)
    if task == 'enhance':
        label = (label == 4).astype(np.float32)
    #-------------------restore if needed, like train enhance--------
    if task != 'all':
        if os.path.exists(os.path.join(check_dir,'u_net_all.txt')):
            saver.restore(sess, os.path.join(check_dir,'u_net_all.ckpt'))
            print('Restore all segmentation as init !')
    if os.path.exists(os.path.join(check_dir, 'u_net_{}.txt'.format(task))):
        saver.restore(sess, os.path.join(check_dir,'u_net_{}.ckpt'.format(task)))
        f = np.loadtxt(os.path.join(check_dir,'u_net_{}.txt'.format(task)))
        step = np.int(f)
        print('Restor task {1} step {0} well.'.format(step, task))
    #---------------------prepare data well------
    print('data process over')
    fake_last_list = []
    for i in range(epoch):   
        num_batch = len(data)//BATCH_SIZE
        print('Each epoch contains {} stps'.format(num_batch))
        loss_list = []
        adverse_list = []
        start_epoch = time.time()

        for j in range(num_batch):
            start = time.time()
            batch_data = data[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            batch_label = label[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            # random crop to [10, 128, 128, 4] for balance the background and features
            random_crop = np.random.randint(low=0, high=52)
            batch_data = batch_data[:, random_crop:(random_crop+128),random_crop:(random_crop+128),:]
            batch_label = batch_label[:, random_crop:(random_crop+128),random_crop:(random_crop+128),:]            
            #-----------start traning-----------
    
            feed_dict = {input_holder: batch_data, label_holder:batch_label}
            sess.run(optimization, feed_dict)
            loss_list.append(sess.run(loss, feed_dict=feed_dict))
            adverse_list.append(sess.run(adverse_loss, feed_dict=feed_dict))
            fake_last_list.append(sess.run(fake_last_layer, feed_dict=feed_dict))
            #----------------if seGAN used--------------
            step += 1 # count
            end = time.time() 
            if step%100 == 0:
                interval = (end-start)*100/60
                print('step {0}, loss {1:.3f}, took {2:.2f} min '.format(step, np.mean(loss_list), interval))
                print('Adverse loss {0:.4f}'.format(np.mean(adverse_list)))
                print('Fake last layer  {:.4f} \n'.format(fake_last_list[-1]))
            if step % 500 == 0:
                saver.save(sess, check_dir+'u_net_{0}.ckpt'.format(task))
                np.savetxt(check_dir+'u_net_{0}.txt'.format(task), [step])
                np.savetxt(check_dir+'fake_indice_{0}.txt'.format(task), fake_last_list)
        end_epoch = time.time()
        print('Task {2} Epoch {0}/{1}'.format(i+1, epoch, task))
        print('Take {:.2f} min'.format((end_epoch-start_epoch)/60))
        print('loss {:.4f} '.format(np.mean(loss_list)))
        print('adverse loss {:.4f}\n'.format(np.mean(adverse_list)))
    return 0
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all', help='all, necrotic, edema, enhance')
    args = parser.parse_args()

    main(args.task)
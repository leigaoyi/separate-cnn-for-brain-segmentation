# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 19:51:56 2018

@author: ky
"""

import tensorflow as tf
#import numpy as np


slim = tf.contrib.slim

def separate_block(inputs, name, channel):
    with tf.variable_scope(name):
#        output_dim = inputs.shape[-1].value
        sconv1 = slim.separable_conv2d(inputs, channel, 3, depth_multiplier=1, stride=1)
        sconv2 = slim.separable_conv2d(sconv1, channel, 3, depth_multiplier=1, stride=1)
    return sconv2


def build_model(inputs, reuse=False):
    '''
    inputs : [BATCH_SIZE, 128, 128, 4]
    separate CNN examples
    '''
    with tf.variable_scope('separate_u_net', reuse=reuse):
        sconv1 = slim.separable_conv2d(inputs, 64, 3, weights_initializer=tf.truncated_normal_initializer(stddev=0.02)
                                       ,depth_multiplier=1, stride=1)
        sconv1 = slim.separable_conv2d(sconv1, 64, 3, weights_initializer=tf.truncated_normal_initializer(stddev=0.02)
                                       ,depth_multiplier=1, stride=1)
        down_conv1 = slim.max_pool2d(sconv1, 3, stride=2, padding='SAME')#[10, 64, 64, 64]
        
        block1 = separate_block(down_conv1, 'block1', 128)
        down_conv2 = slim.max_pool2d(block1, 3, stride=2, padding='SAME') # [10, 32, 32, 128]
        block2 = separate_block(down_conv2, 'block2', 256)
        down_conv3 = slim.max_pool2d(block2, 3, stride=2, padding='SAME')
        block3 = separate_block(down_conv3, 'block3', 512)
        # [10, 16, 16, 512]
        down_conv4 = slim.max_pool2d(block3, 3, stride=2, padding='SAME')
        block3_ex = separate_block(down_conv4, 'block3_extra', 1024)# 8
        
        up_conv1 = slim.conv2d_transpose(block3_ex, 512, 3, stride=2, activation_fn=None)
        concat4 = tf.concat([up_conv1, block3], axis=3)
        block4 = separate_block(concat4, 'block4', 512) #16
        
        up_conv2 = slim.conv2d_transpose(block4, 256, 3, stride=2, activation_fn=None)
        concat3 = tf.concat([up_conv2, block2], axis=3)
        block5 = separate_block(concat3, 'block5', 256) #32
        
        up_conv3 = slim.conv2d_transpose(block5, 128, 3, stride=2, activation_fn=None)
        concat1 = tf.concat([up_conv3, block1], axis=3) #64
        block6 = separate_block(concat1, 'block6', 128)
        
        up_conv4 = slim.conv2d_transpose(block6, 64, 3, stride=2, activation_fn=None) #128
        concat2 = tf.concat([up_conv4, sconv1], axis=3)
        block7 = separate_block(concat2, 'block7', 64)

#        print('block 7 : ', block7.shape)
        conv_out = slim.conv2d(block7, 1, 1, activation_fn= tf.nn.sigmoid) 
        return conv_out
    
def u_net(inputs, reuse=False):
    with tf.variable_scope('u_net', reuse=reuse):
        conv1 = slim.conv2d(inputs, 64, 3)
        conv1 = slim.conv2d(conv1, 64 , 3)#[240, 240]
        down_sample1 = slim.max_pool2d(conv1, 3, padding='SAME')
        
        conv2 = slim.conv2d(down_sample1, 128, 3)
        conv2 = slim.conv2d(conv2, 128, 3)
        down_sample2 = slim.max_pool2d(conv2, 3, padding='SAME')
        
        conv3 = slim.conv2d(down_sample2, 256, 3)
        conv3 = slim.conv2d(conv3, 256, 3)
        down_sample3 = slim.max_pool2d(conv3, 3, padding='SAME')
        
        conv4 = slim.conv2d(down_sample3, 512, 3)
        conv4 = slim.conv2d(conv4, 512, 3)#[30, 30]
        down_sample4 = slim.max_pool2d(conv4, 3, padding='SAME')
        
        conv5 = slim.conv2d(down_sample4, 1024, 3)
        conv5 = slim.conv2d(conv5, 1024, 3)#[15, 15]
        up_sample1 = slim.conv2d_transpose(conv5, 512, 3,stride=2, activation_fn=None)
        
        concat1 = tf.concat([up_sample1, conv4], axis=3)
        conv6 = slim.conv2d(concat1, 512, 3)
        conv6 = slim.conv2d(conv6, 512, 3)
        
        up_sample2 = slim.conv2d_transpose(conv6, 256, 3, stride=2,activation_fn=None)
        concat2 = tf.concat([up_sample2, conv3], axis=3)
        conv7 = slim.conv2d(concat2, 256, 3)
        conv7 = slim.conv2d(conv7, 256, 3)
        
        up_sample3 = slim.conv2d_transpose(conv7, 128, 3, stride=2,activation_fn=None)
        concat3 = tf.concat([up_sample3, conv2], axis=3)
        conv8 = slim.conv2d(concat3, 128, 3)
        conv8 = slim.conv2d(conv8, 128, 3)
        
        up_sample4 = slim.conv2d_transpose(conv8, 64, 3,stride=2, activation_fn=None)
        concat4 = tf.concat([up_sample4, conv1], axis=3)
        conv9 = slim.conv2d(concat4, 64, 3)
        conv9 = slim.conv2d(conv9, 64, 3)
        
        conv10 = slim.conv2d(conv9, 1, 1, activation_fn=tf.nn.sigmoid)
        return conv10
        
def critic(inputs, reuse=False):
    
    return 0


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [10, 128, 128, 4])        
    predict = u_net(x)
    print(predict.shape)
    total_parameters = 0
    for variable in tf.trainable_variables():
    # shape is an array of tf.DimensiWon
        shape = variable.get_shape()
#        print(shape)
#        print(len(shape))
        variable_parameters = 1
        for dim in shape:
#            print(dim)
            variable_parameters *= dim.value
#        print(variable_parameters)
        total_parameters += variable_parameters
    print('total number : ',total_parameters)
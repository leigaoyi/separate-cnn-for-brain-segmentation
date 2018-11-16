# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:45:34 2018

@author: kasy
"""

import tensorflow as tf
#import numpy as np

def mask_seg_input(input_img, seg):
    '''
    inputs : [10, 128, 128, 4]
    seg : [10, 128, 128, 1]
    '''
    mask_list = []
    for i in range(4):
        mask_list.append(input_img[..., i]*seg[:,:,:,0])
    mask_arr = tf.stack(mask_list, axis=3)
    return mask_arr

#input_ = tf.placeholder(tf.float32, [10, 128, 128, 4])
#seg_ = tf.placeholder(tf.float32, [10, 128, 128, 1])
#mask_ = mask_seg_input(input_, seg_)
#print(mask_.shape)
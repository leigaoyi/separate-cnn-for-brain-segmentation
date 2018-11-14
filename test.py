# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:17:57 2018

@author: kasy
"""

import tensorflow as tf
import numpy as np
import glob
import os
import nibabel as nib

from model_separate import build_model

task = 'enhance'
test_type = 'half' #small, half, all
save_dir = 'checkpoints'
save_hgg_path = './result/{0}/HGG/'.format(str(task))
save_lgg_path = './result/{0}/LGG/'.format(str(task))
data_path = './data/MICCAI_BraTS17_Data_Training/HGG/' # ubuntu data path
#data_path = './data/MICCAI_BraTS17_Data_Training_IPP/MICCAI_BraTS17_Data_Training/HGG/' #windows
batch_size = 10
HGG_data_path = "data/MICCAI_BraTS17_Data_Training/HGG"
LGG_data_path = "data/MICCAI_BraTS17_Data_Training/LGG"
#HGG_data_path = 'data/temp/HGG'
#LGG_data_path = 'data/temp/LGG'



#===================load data name list ==========
def load_folder_list(path=""):
    """Return a folder list in a folder by given a folder path.
    Parameters
    ----------
    path : str
        A folder path.
    """
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

if test_type == 'all':
    HGG_path_list = load_folder_list(path=HGG_data_path)
    LGG_path_list = load_folder_list(path=LGG_data_path)
elif test_type == 'half':
    HGG_path_list = load_folder_list(path=HGG_data_path)[150:]# DEBUG WITH SMALL DATA
    LGG_path_list = load_folder_list(path=LGG_data_path)[50:] # DEBUG WITH SMALL DATA
elif test_type == 'small':
    HGG_path_list = load_folder_list(path=HGG_data_path)[0:2] # DEBUG WITH SMALL DATA
    LGG_path_list = load_folder_list(path=LGG_data_path)[0:1] # DEBUG WITH SMALL DATA
else:
    exit("Unknow DATA_SIZE")
print(len(HGG_path_list), len(LGG_path_list)) #210 #75

HGG_name_list = [os.path.basename(p) for p in HGG_path_list]
LGG_name_list = [os.path.basename(p) for p in LGG_path_list]

data_types = ['flair', 't1', 't1ce', 't2']

# read the files
if not os.path.exists(save_hgg_path):
    os.makedirs(save_hgg_path)
    os.makedirs(save_lgg_path)


# test hgg data and lgg data

#--------------------begin sess environment-------------
sess = tf.Session()
x = tf.placeholder(tf.float32, [batch_size, 128, 128, 4])
y = tf.placeholder(tf.float32, [batch_size, 128, 128, 1])
model_test = build_model(x, reuse=False)
pre_model_name = './checkpoints/u_net_{0}.ckpt'.format(task)
pre_step_num = './checkpoints/u_net_{0}.txt'.format(task)
init = tf.global_variables_initializer()
sess.run(init)
#-------------------restor parameters-------------------
saver = tf.train.Saver()
saver.restore(sess, pre_model_name)
step = int(np.loadtxt(pre_step_num))
print('Restore well')

def produce_batch_seg(input_data):
    '''
    input_data : [10, 180, 180, 4]
    output : [10, 180, 180, 1] (segment from model)
    '''
    left_up, left_down = input_data[:, :128, :128, :], input_data[:, :128, 52:, :]
    right_up, right_down = input_data[:, 52:, :128, :], input_data[:, 52:, 52:, :]
    seg_out = np.zeros((batch_size, 180, 180, 1))
    
    feed_dict_1 = {x:left_up}
    feed_dict_2 = {x:left_down}
    feed_dict_3 = {x:right_up}
    feed_dict_4 = {x:right_down}
    
    seg_out[:, :128, :128, :] = sess.run(model_test, feed_dict=feed_dict_1)
    seg_out[:, :128, 52:, :] = sess.run(model_test, feed_dict=feed_dict_2)
    seg_out[:, 52:, :128, :] = sess.run(model_test, feed_dict=feed_dict_3)
    seg_out[:, 52:, 52:, :] = sess.run(model_test, feed_dict=feed_dict_4)
    return seg_out

def produce_whole_seg(input_path, file_name):
    '''
    input single MRI path
    '''
    temp_data = []
    train_data = []
    whole_seg = []
    seg_out = np.zeros([180, 180, 155], dtype=np.float32)
    for i in data_types:
        MRI_PATH = os.path.join(input_path, file_name, file_name+'_{}.nii.gz'.format(i))
        img = nib.load(MRI_PATH).get_data()[30:210, 30:210, :]
        mean = np.mean(img)
        std = np.std(img)
        img = (img-mean)/std
        temp_data.append(img)
    for i in range(temp_data[0].shape[2]):
        cat = np.stack([temp_data[0][:,:,i], temp_data[1][:,:,i],
                        temp_data[2][:,:,i], temp_data[2][:,:,i]], axis=2)
        train_data.append(cat)
    train_data = np.asarray(train_data)
    for i in range(15):
        batch_input = train_data[10*i:(10*i+10), ...]
        batch_seg = produce_batch_seg(batch_input)
        whole_seg.append(batch_seg)
            
    batch_input = train_data[-10:, ...]
    batch_seg = produce_batch_seg(batch_input)
    whole_seg.append(batch_seg[5:, ...])
    whole_seg = np.concatenate(whole_seg, axis=0)
    for i in range(155):
        seg_out[:,:, i] = np.squeeze(whole_seg[i, ...])
    seg_out = np.array(seg_out)
    seg_out = np.squeeze(seg_out)
#    print('seg_out shape ',seg_out.shape)
    gt_path = os.path.join(input_path, file_name, file_name+'_seg.nii.gz')
    gt_img = nib.load(gt_path).get_data()[30:210, 30:210, :]
    gt_affine = nib.load(gt_path).get_affine()
#    print('groud truth shape ', gt_img.shape)
    if task == 'all':
        gt_img = (gt_img>0).astype(np.float32)
    elif task == 'enhance':
        gt_img = (gt_img==4).astype(np.float32)
    elif task == 'edema' :
        gt_img = (gt_img == 2).astype(np.float32)
    elif task == 'necrotic':
        gt_img = (gt_img == 1).astype(np.float32)
    else:
        assert False, 'task wrong'
    
    return seg_out, gt_img, gt_affine

def dice_coe(output, target, loss_type='jaccard', axis=(0, 1, 2), smooth=1e-5):
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

def store_MRI(path, seg_img, seg_affine):
    fig = np.zeros([240, 240, 155], dtype=np.float32)
    fig[30:210, 30:210, :] = seg_img
    fig = (fig>0.2).astype(np.int8)
    fig_nii = nib.Nifti1Image(fig, affine=seg_affine)
    nib.save(fig_nii, path)
    return 0


#----------------testing dice score-------------
hgg_dice_list = []
lgg_dice_list = []
print('Prepare HGG test ')
for i in range(len(HGG_name_list)):
    dir_path = HGG_data_path
    file_path = HGG_name_list[i]
    save_path = os.path.join(save_hgg_path, file_path+'_seg.nii.gz')
    seg, gt, affine = produce_whole_seg(dir_path, file_path)
    store_MRI(save_path, seg, affine)
    hgg_dice_list.append(sess.run(dice_coe(seg, gt)))
    if i%(len(HGG_name_list)//4) == 0:
        print('Testing {:.1f}%'.format(i/len(HGG_name_list)*100))
np.savetxt('./result/{}_hgg_dice.txt'.format(task), hgg_dice_list)

print('Prepare LGG test ')
for i in range(len(LGG_name_list)):
    dir_path = LGG_data_path
    file_path = LGG_name_list[i]
    save_path = os.path.join(save_lgg_path, file_path+'_seg.nii.gz')
    seg, gt, affine = produce_whole_seg(dir_path, file_path)
    store_MRI(save_path, seg, affine)
    lgg_dice_list.append(sess.run(dice_coe(seg, gt)))
    if i%(len(LGG_name_list)//4) == 0:
        print('Testing {:.1f}%'.format(i/len(HGG_name_list)*100))
np.savetxt('./result/{}_lgg_dice.txt'.format(task), lgg_dice_list)
print('task : {}'.format(task))
print('step : {}'.format(step))
print('HGG cases {0}, LGG cases {1}'.format(len(HGG_name_list), len(LGG_name_list)))
print('HGG dice mean {:.4f}'.format(np.mean(hgg_dice_list)))
print('HGG dice median {:.4f}'.format(np.median(hgg_dice_list)))
print('LGG dice mean {:.4f}'.format(np.mean(lgg_dice_list)))
print('LGG dice median {:.4f}'.format(np.median(lgg_dice_list)))
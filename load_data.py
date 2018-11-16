# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:43:25 2018

@author: ky
"""
import os
#import tensorflow as tf
import numpy as np
import nibabel as nib
#import pickle

#======================Setting================
DATA_SIZE = 'half' # (small, half or all)
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

if DATA_SIZE == 'all':
    HGG_path_list = load_folder_list(path=HGG_data_path)
    LGG_path_list = load_folder_list(path=LGG_data_path)
elif DATA_SIZE == 'half':
    HGG_path_list = load_folder_list(path=HGG_data_path)[0:120]# DEBUG WITH SMALL DATA
    LGG_path_list = load_folder_list(path=LGG_data_path)[0:40] # DEBUG WITH SMALL DATA
elif DATA_SIZE == 'small':
    HGG_path_list = load_folder_list(path=HGG_data_path)[0:2] # DEBUG WITH SMALL DATA
    LGG_path_list = load_folder_list(path=LGG_data_path)[0:1] # DEBUG WITH SMALL DATA
else:
    exit("Unknow DATA_SIZE")
print(len(HGG_path_list), len(LGG_path_list)) #210 #75

HGG_name_list = [os.path.basename(p) for p in HGG_path_list]
LGG_name_list = [os.path.basename(p) for p in LGG_path_list]

data_types = ['flair', 't1', 't1ce', 't2']
#with open('mean_std_dict.pickle', 'rb') as f:
#    mean_std = pickle.load(f)

def create_input_data():
    '''
    crop into [180, 180, 128]
    '''
    seg_collection = []
    data_collection = []
    for i in range(len(HGG_name_list)):
        temp_data = []
        for j in data_types:
            img_path = os.path.join(HGG_path_list[i],HGG_name_list[i]+'_{}.nii.gz'.format(j)) 
            img = nib.load(img_path).get_data()[30:210, 30:210, 13:141]
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean)/std
            temp_data.append(img)
        #---------stack flair, t1, t1ce, t2---------
        label_path = os.path.join(HGG_path_list[i], HGG_name_list[i]+'_seg.nii.gz')
        label = nib.load(label_path).get_data()[30:210, 30:210, 13:141]
        for j in range(temp_data[0].shape[2]):
            data_collection.append(np.stack([temp_data[0][:,:,j],
                                         temp_data[1][:,:,j],
                                         temp_data[2][:,:,j],
                                         temp_data[3][:,:,j]],axis=2))
            seg_collection.append(label[:,:,j])
    
    for i in range(len(LGG_name_list)):
        temp_data = []
        for j in data_types:
            img_path = os.path.join(LGG_path_list[i],LGG_name_list[i]+'_{}.nii.gz'.format(j)) 
            img = nib.load(img_path).get_data()[30:210, 30:210, 13:141]
            mean = np.mean(img)# mean and std
            std = np.std(img)
            img = (img - mean)/std
            temp_data.append(img)
        #---------stack flair, t1, t1ce, t2---------
        label_path = os.path.join(LGG_path_list[i], LGG_name_list[i]+'_seg.nii.gz')
        label = nib.load(label_path).get_data()[30:210, 30:210, 13:141]
        for j in range(temp_data[0].shape[2]):
            data_collection.append(np.stack([temp_data[0][:,:,j],
                                         temp_data[1][:,:,j],
                                         temp_data[2][:,:,j],
                                         temp_data[3][:,:,j]],axis=2))
            seg_collection.append(label[:,:,j])
    data_collection = np.asarray(data_collection, dtype=np.float32)
    seg_collection = np.asarray(seg_collection)
    #-------------center crop------------
    # (240, 240) --->>  (180, 180)
    return data_collection, seg_collection[:,:,:,np.newaxis]



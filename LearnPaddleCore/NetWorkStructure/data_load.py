import paddle
import paddle.fluid as fluid
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import gzip
import json

# Define data reader
def data_load(mode='train', batch_size=100, img_row=28, img_col=28):
    # Read data
    file_path = './work/mnist_json.gz'
    data = json.load(gzip.open(file_path))
    # Separate data set
    train_set, val_set, eval_set = data

    # Get imgs and label from dataset
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'evaluate':
        imgs = eval_set[0]
        labels = eval_set[1]

    # Create index list
    index_list = list(range(len(imgs)))

    # Define data generator
    def data_generator():
        # If in train mode, shuffle the index for training
        if mode == 'train':
            np.random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            img = np.reshape(imgs[i], [1, img_row, img_col]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('float32')
            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list) == batch_size:
                yield np.array(imgs_list), np.array(labels_list)
                imgs_list = []
                labels_list = []
            
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator
        

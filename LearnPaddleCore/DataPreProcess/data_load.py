# MNIST dataset
# data - train_set - train_images - shape: [5000, 28 * 28]
#                  - train_labels - shape: [5000,]
#      - val_data
#      - test_data  

import os
import json
import gzip
import data_check
import numpy as np

def load_data(mode='train'):
    file_path = './work/mnist_json.gz'
    print('loading mnist dataset from {}'.format(file_path))
    # load data from json file 
    json_data = json.load(gzip.open(file_path))
    print('mnist dataset load done')
    # Separate dataset
    train_set, val_set, test_set = json_data

    if mode=='train':
        imgs, label = train_set[0], train_set[1]
    elif mode=='val':
        imgs, label = val_set[0], val_set[1]
    elif mode=='test':
        imgs,label = test_set[0], test_set[1]
    else:
        raise Exception("Input mode is illegal, it should be one of ['train', 'val', 'test']")

    print('Num of train dataset: {}'.format(len(imgs)))

    return imgs, label

def data_load(mode='train'):
    BATCH_SIZE=1000
    IMG_ROW=28
    IMG_COL=281

    imgs, labels = load_data(mode)
    data_check.data_check(imgs, labels)
    # Get length of dataset
    data_len = len(imgs)
    index_list = list(range(data_len))

    def data_generator():
        if mode == 'train':
            # Shuffle the index list
            np.random.shuffle(index_list)
            
        img_list = []
        label_list = []
        for i in index_list:
            img = np.reshape(imgs[i], [1, IMG_ROW, IMG_COL]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('float32')
            img_list.append(img)
            label_list.append(label)
            if len(img_list) == BATCH_SIZE:
                yield np.array(img_list), np.array(label_list)
                img_list = []
                label_list = []

        if len(img_list) > 0:
            yield np.array(img_list), np.array(label_list)
        
    return data_generator
    



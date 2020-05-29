import numpy as np
import os
import json
import gzip

IMG_ROW = 28
IMG_COL = 28
BATCH_SIZE = 1000

'''
def data_generator(index_list, imgs, label, row, col, batch_size=10):
    imgs_list = []
    label_list = []
    for i in index_list:
        # Transform data into desired format
        img = np.reshape(imgs[i], [1, row, col]).astype('float32')
        label = np.reshape(label[i], [1]).astype('float32')
        imgs_list.append(img)
        label_list.append(label)

        if len(imgs_list) == batch_size:
            # Generate a dataset with size of ordered batch 
            yield np.array(imgs_list), np.array(label_list)
            # clear temp data list
            imgs_list = []
            label_list = []

    if len(imgs_list) > 0:
        yield np.array(imgs_list), np.array(label_list)
    return data_generator
'''
file_path = './work/mnist_json.gz'
print('loading mnist dataset from {}'.format(file_path))
# load data from json file 
json_data = json.load(gzip.open(file_path))

# Separate dataset
train_set, val_set, test_set = json_data

# Print train dataset information
imgs, labels = train_set[0], train_set[1]

imgs_length = len(imgs)
# Generate a list of data sequence number
index_list = list(range(imgs_length))
batch_size = 100
print('Print index list before shuffle: {}'.format(index_list))
np.random.shuffle(index_list)
print('Print index list after shuffle: {}'.format(index_list))

def data_generator():
    imgs_list = []
    label_list = []
    for i in index_list:
        # Transform data into desired format
        img = np.reshape(imgs[i], [1, IMG_ROW, IMG_COL]).astype('float32')
        label = np.reshape(labels[i], [1]).astype('float32')
        imgs_list.append(img)
        label_list.append(label)

        if len(imgs_list) == BATCH_SIZE:
            yield np.array(imgs_list), np.array(label_list)
            imgs_list = []
            label_list = []

    if len(imgs_list) > 0:
        yield np.array(imgs_list), np.array(label_list)
    return data_generator

# Declare a data loader
train_loader = data_generator
# Read data
for batch_i, data in enumerate(train_loader()):
    image_data, label_data = data
    if batch_i == 0:
        print('Print the shape of first batch data')
        print('Shape of image: {}, Shape of label: {}'.format(image_data.shape, label_data.shape))
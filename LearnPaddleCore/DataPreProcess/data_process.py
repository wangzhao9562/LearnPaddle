# MNIST dataset
# data - train_set - train_images - shape: [5000, 28 * 28]
#                  - train_labels - shape: [5000,]
#      - val_data
#      - test_data  

import os
import json
import gzip

file_path = './work/mnist_json.gz'
print('loading mnist dataset from {}'.format(file_path))
# load data from json file 
json_data = json.load(gzip.open(file_path))
print('mnist dataset load done')
# Separate dataset
train_set, val_set, test_set = json_data

# Define image size format
IMG_ROW = 28
IMG_COL = 28

# Print train dataset information
imgs, label = train_set[0], train_set[1]
print('Num of train dataset: {}'.format(len(imgs)))

# Print valuate dataset information
imgs, label = val_set[0], val_set[1]
print('Num of valuate dataset: {}'.format(len(imgs)))

# Print test dataset information
imgs,label = test_set[0], test_set[1]
print('Num of test dataset: {}'.format(len(imgs)))


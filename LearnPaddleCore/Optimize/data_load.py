import numpy as np
import gzip
import json

def data_load(mode='train'):
    file_path = './work/mnist_json.gz'
    data = json.load(gzip.open(file_path))

    # Separate the dataset
    train_set, val_set, eval_set = data

    IMG_ROW = 28
    IMG_COL = 28

    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'value':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'evaluate':
        imgs = eval_set[0]
        labels = eval_set[1]

    img_len = len(imgs)

    index_list = list(range(img_len))

    BATCH_SIZE = 100

    # Define the data generator
    def data_generator():
        if mode == 'train':
            np.random.shuffle(index_list)
        imgs_list = []
        labels_list = []

        for i in index_list:
            img = np.reshape(imgs[i], [1, IMG_ROW, IMG_COL]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list) == BATCH_SIZE:
                yield np.array(imgs_list), np.array(labels_list)
                imgs_list = []
                labels_list = []

        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator


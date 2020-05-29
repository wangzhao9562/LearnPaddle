import os
import json
import gzip
import numpy as np
import data_load

import paddle
import paddle.fluid as fluid

# Load data in asynchronous way
place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    train_loader = data_load.data_load(mode='train')
    data_loader = fluid.io.DataLoader.from_generator(capacity=5, return_list=True)
    # Set data generator
    data_loader.set_batch_generator(train_loader, places=place)
    # Print data shape
    for i, data in enumerate(data_loader):
        image_data, label_data = data
        print(i, image_data.shape, label_data.shape)
        if i>= 5:
            break
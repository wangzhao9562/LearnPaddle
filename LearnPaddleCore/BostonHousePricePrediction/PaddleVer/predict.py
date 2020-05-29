# import paddle core lib: paddle/fluid
# import dynamic graph lib: paddle/fluid/dygraph
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
import os
import random

import train_model

def load_test_data(data_dir):
    f = open(data_dir, 'r')
    datas = f.readlines()
    # Select test data
    test_data = datas[-10]
    test_data = test_data.strip().split()
    data = [float(ele) for ele in test_data]

    # Normalization
    for i in range(len(data) - 1):
        data[i] = (data[i] - train_model.avg_values[i]) / (train_model.max_values[i] - train_model.min_values[i])

    data = np.reshape(np.array(data[:-1]), [1, -1]).astype(np.float32)
    label = data[-1]
    return data, label

with dygraph.guard():
    model_dict, _ = fluid.load_dygraph('LR_model')
    model = train_model.Regressor("Regressor")
    model.load_dict(model_dict)
    # In evaluate mode
    model.eval()

    test_data, label = load_test_data('./work/housing.data')
    # Transform data into dygraph variable
    test_data = dygraph.to_variable(test_data)
    results = model(test_data)

    # Denormalization
    results = results * (train_model.max_values[-1] - train_model.min_values[-1] + train_model.avg_values[-1])
    print("Interface result is {}, the corresponding label is {}".format(results.numpy(), label))
import paddle 
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import numpy as np
import os
from PIL import Image

def load_data(batch_size = 8):
    trainset = paddle.dataset.mnist.train()
    train_reader = paddle.batch(trainset, batch_size)
    return train_reader

def load_image(file_path):
    # Transform image into gray
    im = Image.open(file_path).convert('L')
    print(np.array(im))
    im = im.resize((28,28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, -1).astype(np.float32)
    # Normalization
    im = 1 - im / 127.5
    return im
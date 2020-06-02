import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np

# Definition of network
class MnistNetwork(fluid.dygraph.Layer):
    def __init__(self):
        super(MnistNetwork, self).__init__()
    
        # Define a convolution layer
        self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # Define a pool layer
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # Define a convolution layer
        self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # Define a pool layer
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # Define a full-connect layer
        self.fc = Linear(input_dim=980, output_dim=10, act='softmax')

    def forward(self, input):
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = fluid.layers.reshape(x, [x.shape[0], 980])
        x = self.fc(x)        
        return x

    

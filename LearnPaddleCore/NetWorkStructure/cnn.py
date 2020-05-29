# CNN includes convolution layers and pool layers
# -- Convolution layers scan input and generate descrption with more abstract features
# -- Pool layers filters these features, reserved the important informations

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as py

class MnistCNN(fluid.dygraph.Layer):
    def __init__(self):
        super(MnistCNN, self).__init__()
        
        # Defines coovolution layer
        # Input of convolution layer: 
        #   -- num_channels: Number of input features
        #   -- num_filters: Number of output features
        #   -- filter_size: Size of convolution core
        #   -- stride: Step of convolution 
        #   -- padding: Padding number of residule nodes
        self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # Defines pool layer
        # Input of pool layer:
        #   -- pool_size: Size of pool core
        #   -- pool_stride: Stride of pool 
        #   -- pool_type: Pool method
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # Defines output full connect layer
        self.fc = Linear(input_dim=980, output_dim=1, act=None)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x
 


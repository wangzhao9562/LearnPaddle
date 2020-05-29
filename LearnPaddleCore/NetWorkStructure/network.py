# Classical neural network includes four layers:
#   -- One input layer
#   -- Two hidden layer
#   -- One output layer
#   -- Add sigmoid function into hidden layer to improve the nonlinear process ability of network
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np

def sigmoid(x):
    return 1. /(1. + np.exp(-x))

class MnistNetwork(fluid.dygraph.Layer):
    def __init__(self):
        super(MnistNetwork, self).__init__()
        # Define two full connective layers
        self.fc1 = Linear(input_dim=784, output_dim=10, act='sigmoid')
        self.fc2 = Linear(input_dim=10, output_dim=10, act='sigmoid')
        # Define output layer
        self.fc3 = Linear(input_dim=10, output_dim=1, act=None)

    # Define the forward computation of network
    def forward(self, input, label=None):
        inputs = fluid.layers.reshape(input, [input.shape[0], 784]) 
        output1 = self.fc1(inputs)
        output2 = self.fc2(output1)
        output = self.fc3(output2)
        return output


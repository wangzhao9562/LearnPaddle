import paddle 
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import numpy as np
import os

# Define minist data recognition network 
class MnistRec(fluid.dygraph.Layer):
    def __init__(self, dim):
        super(MnistRec, self).__init__()

        # Define a full connect layer
        self.fc = Linear(dim * dim, output_dim = 1, act = None)

    # Define forward compute process for model
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs
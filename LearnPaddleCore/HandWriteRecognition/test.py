import paddle 
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import numpy as np
import os
import model
import load_data
import matplotlib.image as mping
import matplotlib.pyplot as plt

# Read image 
test_img = mping.imread('./work/example_0.png')
# Show image
plt.imshow(test_img)
plt.show()

# Process of predict
with fluid.dygraph.guard():
    mnist_model = model.MnistRec(28)
    # load model
    model_dict, _ = fluid.load_dygraph('./mnist_model/mnist_model')
    mnist_model.load_dict(model_dict)
    # Input data
    mnist_model.eval()
    tensor_img = load_data.load_image('./work/example_0.png')
    result = mnist_model(fluid.dygraph.to_variable(tensor_img))

    # Predict
    print("The predicted number is: {}", result.numpy().astype('int32'))
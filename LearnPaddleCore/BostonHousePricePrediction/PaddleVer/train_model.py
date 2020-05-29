# import paddle core lib: paddle/fluid
# import dynamic graph lib: paddle/fluid/dygraph
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
import os
import random

def load_data():
    # Load data from file
    data_dir = './work/housing.data'
    data = np.fromfile(data_dir, sep=' ')

    # Each data has 14 items, 1st to 13th are factors, the 14th represents the mid price of house pride
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # Reshap data into [N, 14]
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # Compute maximum, minimum, average of data
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), training_data.sum(axis=0) / training_data.shape[0]
  
    global max_values
    max_values = maximums
    global min_values
    min_values = minimums
    global avg_values
    avg_values = avgs

    # Normalization 
    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
        
    # Separate training data and evaluating data
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

class Regressor(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Regressor, self).__init__(name_scope)
        name_scope = self.full_name()
        # Create a linear layer
        self.fc = Linear(input_dim=13, output_dim=1, act=None)

    # Forward computation function
    def forward(self, input):
        x = self.fc(input)
        return x

# Define work environment for paddle dynamic graph
with fluid.dygraph.guard():
    # Declare the predefined linear model
    model = Regressor("Regressor")
    # Start train mode
    model.train()
    # load data
    train_data, test_data = load_data()
    # Set SGD optimizer
    opt = fluid.optimizer.SGD(learning_rate=0.01, parameter_list=model.parameters())

with dygraph.guard(fluid.CPUPlace()):
    EPOCH_NUM = 10
    BATCH_SIZE = 10

    for epoch_i in range(EPOCH_NUM):
        # shuffle the data
        np.random.shuffle(train_data)
        # Separate data
        mini_batches = [train_data[k: k+BATCH_SIZE] for k in range(0, len(train_data), BATCH_SIZE)]
        for iter_i, mini_batch in enumerate(mini_batches):
            x = np.array(mini_batch[:, :-1]).astype('float32')
            y = np.array(mini_batch[:, -1:]).astype('float32')
            # Trasform numpy data into paddle variable
            house_features = dygraph.to_variable(x)
            prices = dygraph.to_variable(y)

            # Forward compute
            predicts = model.forward(house_features)

            # Compute MSN loss
            loss = fluid.layers.square_error_cost(predicts, label=prices)
            avg_loss = fluid.layers.mean(loss)
            if iter_i % 20 == 0:
                print("epoch: {}, iter: {}, loss is : {}".format(epoch_i, iter_i, avg_loss.numpy()))

            # Backward broadcast
            avg_loss.backward()
            # Minimize loss, update weights
            opt.minimize(avg_loss)
            # Clear the gradient
            model.clear_gradients()

    # Save the model
    fluid.save_dygraph(model.state_dict(), 'LR_model')
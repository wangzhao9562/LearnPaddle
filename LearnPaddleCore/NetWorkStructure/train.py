import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np
import network
import data_load

with fluid.dygraph.guard():
    model = network.MnistNetwork()
    # Set model as train mode
    model.train()
    
    # load data
    data_loader = data_load.data_load('train')

    # Define optimizer
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())

    epoch_time = 5

    # Train
    for epoch_id in range(epoch_time):
        for batch_id, data in enumerate(data_loader()):
            # Transform data
            img_data, label_data = data
            img = fluid.dygraph.to_variable(img_data)
            label = fluid.dygraph.to_variable(label_data)

            predict = model.forward(img)

            # Compute loss
            loss = fluid.layers.square_error_cost(predict, label)
            avg_loss = fluid.layers.mean(loss)

            # Print messages
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is : {}".format(epoch_id, batch_id, avg_loss.numpy()))

            avg_loss.backward()
            optimizer.minimize(avg_loss)
            optimizer.clear_gradients()

    # Save model
    fluid.save_dygraph(model.state_dict(), './classic/mnist')

            
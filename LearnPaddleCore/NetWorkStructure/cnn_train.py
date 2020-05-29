import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np
import network
import data_load
import cnn

with fluid.dygraph.guard():
    # Create cnn model
    model = cnn.MnistCNN()
    model.train()

    # Load data
    data_loader = data_load.data_load()

    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())

    epoch_num = 5

    # Train
    for epoch_id in range(epoch_num):
        for batch_id, data in enumerate(data_loader()):
            # Get images and labels
            img_data, label_data = data
            img = fluid.dygraph.to_variable(img_data)
            label = fluid.dygraph.to_variable(label_data)

            # Forward compute
            predict = model.forward(img)

            # Compute loss
            loss = fluid.layers.square_error_cost(predict, label)

            avg_loss = fluid.layers.mean(loss)

            # Print information
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

            # Backward broadcast
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            optimizer.clear_gradients()

    fluid.save_dygraph(model.state_dict(), 'cnn_mnist\\cnn_mnist')
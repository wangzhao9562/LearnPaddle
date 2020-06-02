import paddle
import paddle.fluid as fluid
import os
import numpy as np
import data_load
import network

with fluid.dygraph.guard():
    model = network.MnistNetwork()
    model.train()

    # load data
    train_loader = data_load.data_load(mode='train')
    
    # Define optimizer 
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())

    EPOCH_TIME=5

    for epoch_id in range(EPOCH_TIME):
        for batch_id, data in enumerate(train_loader()):
            image_data, label_data = data
            img = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)

            # Forward computation
            predict = model.forward(img)

            # Compute loss
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)

            if batch_id % 200 == 0:
                print('epoch is: {}, batch is: {}, loss is: {}'.format(epoch_id, batch_id, avg_loss.numpy()))

            # Optimize parameters
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    # Save model parameters
    fluid.save_dygraph(model.state_dict(), './mnist_model/mnist')

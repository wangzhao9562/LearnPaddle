import paddle 
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import numpy as np
import os
import model
import load_data

# Define paddle dynamic graph work environment
with fluid.dygraph.guard():
    # Create model
    mnist_model = model.MnistRec(28) # pixel dimension of image is 28
    # Start train mode
    mnist_model.train()
    # Define data reader
    # train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=16)
    train_reader = load_data.load_data(batch_size=16)
    # Define optimizer
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=mnist_model.parameters())

    print("train_reader: {}".format(train_reader))

    # Two nest circular layer in train process
    EPOCH_NUM = 10
    for epoch_i in range(EPOCH_NUM):
        for batch_i, data in enumerate(train_reader()):
            image_data = np.array([x[0] for x in data]).astype('float32')
            label_data = np.array([x[1] for x in data]).astype('float32').reshape(-1, 1)

            # print("batch id: {}".format(batch_i))

            # Transform data into form in paddle dynamic graph
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)

            # Forward computation
            predict = mnist_model(image)

            # Compute loss 
            loss = fluid.layers.square_error_cost(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            if batch_i != 0 and batch_i % 1000 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_i, batch_i, avg_loss.numpy()))
            
            # Backward broadcast
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            mnist_model.clear_gradients()

    fluid.save_dygraph(mnist_model.state_dict(), './mnist_model')
    


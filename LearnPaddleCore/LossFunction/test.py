import paddle
import paddle.fluid as fluid
import numpy as np
import data_load
import network

with fluid.dygraph.guard():
    model = network.MnistNetwork()
    # Load model
    model_dict, _ = fluid.load_dygraph('./mnist_model/mnist.pdparams')
    model.load_dict(model_dict)
    model.eval()

    # load data
    eval_loader = data_load.data_load(mode='evaluate')

    for batch_id, data in enumerate(eval_loader()):
        img_data, label_data = data
        img = fluid.dygraph.to_variable(img_data)
        label = fluid.dygraph.to_variable(label_data)

        # Get evaluate of model
        result = model(img)

        lab = np.argsort(result.numpy())
        
        f# print('batch: {}, img: {}'.format(batch_id, img))
        print('batch: {}, result: {}, practical: {}'.format(batch_id, lab[0][-1], label[0][-1]))

        
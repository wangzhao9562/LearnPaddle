import load_data
import paddle 
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

train_reader = load_data.load_data(batch_size = 8)
for batch_id, data in enumerate(train_reader()):
    # Get image data, transform it into array in type of float32
    img_data = np.array([x[0] for x in data]).astype('float32')
    # Get image label, transfrom it into array in type of float32
    label_data = np.array([x[1] for x in data]).astype('float32')
    # Print data shape
    # shape is (8, 784), 8 is batch_size, 784 = 28 * 28 represents the pixel size of image
    print("Shape of image data and responding data is: ", img_data.shape, img_data[0])
    print("Shape of image label and responding data is: ", label_data.shape, label_data[0])
    break

print("\nPrint the first imgage in first batch, responding data is {}".format(label_data[0]))
# Print the first image in first batch
img = np.array(img_data[0] + 1) * 127.5
img = np.reshape(img, [28, 28]).astype(np.uint8)

plt.figure("Image")
plt.imshow(img)
plt.axis('on') # open axis
plt.title('imgae') # open title
plt.show()


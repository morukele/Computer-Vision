# CNN modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Visiualization modules
import cv2
import matplotlib.pyplot as plt

# Numerical Modules
import numpy as np

# Defining a neural network with a single convolutional layer with four filters


class Net(nn.Module):

    # define the layer of the network
    def __init__(self, weight):
        super(Net, self).__init__()
        # Initializing the weights of the convolutional layer to be weights of the
        # 4 defined filters
        k_height, k_width = weight.shape[2:]

        # applying the four grayscale filters
        # this convo layer is expected to return an output of 4 images
        self.conv = nn.Conv2d(1, 4, kernel_size=(
            k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

        # Defining the pooling later
        self.pool = nn.MaxPool2d(4, 4)

    # define the feedforward behavior
    # x defines the inout image tensor
    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre and post activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        # Applying the pooling layer
        pooled_x = self.pool(activated_x)

        # returns all layers
        return conv_x, activated_x, pooled_x


# Setting the image
im_path = 'images/udacity_sdc.png'

# Loading the image and converting to grayscale
im = cv2.imread(im_path)
im_gray = cv2.cvtColor(im, cv2. COLOR_BGR2GRAY)

# Normalizing the image
im_gray = im_gray.astype("float32")/255

# Plotting the image
plt.imshow(im_gray, cmap='gray')
plt.show()

# Defining and visualizing the filters
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1],
                        [-1, -1, 1, 1], [-1, -1, 1, 1]])


# Defining four different filters as linear combination of 'filter_vals'
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])

# instantiating the model and setting the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)

# Creating an helper function for visualizing the output of a given layer


def viz_layer(layer, n_filters=4, title='Missing Title'):
    fig = plt.figure(figsize=(20, 20))
    plt.title(title)

    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))


# Ploting the original image
plt.imshow(im_gray, cmap='gray')
plt.show()

# visualizing all filters
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0, right=1.5, bottom=0.8,
                    top=1, hspace=0.05, wspace=0.05)
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
plt.show()

# converting the image into an input Tensor
gray_im_tensor = torch.from_numpy(im_gray).unsqueeze(0).unsqueeze(1)

# getting the convolutional layer (pre and post activation)
conv_layer, activated_layer, pooled_layer = model(gray_im_tensor)

# visualzing the output results
viz_layer(conv_layer, title='Convoluation Layer')
viz_layer(activated_layer, title='Activated Layer')
viz_layer(pooled_layer, title='Pooled layer')
plt.show()

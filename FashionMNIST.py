# Neural Network Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Visualization modules
import numpy as np
import matplotlib.pyplot as plt

# Loading and Transforming Data
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

# Transforming the image data sets to Tensors

# Defining a transform to read the data in as a tensor
data_transform = transforms.ToTensor()

# choosing the training and test datasets
train_data = FashionMNIST(root='./data', train=True,
                          download=True, transform=data_transform)

test_data = FashionMNIST(root='./data', train=False,
                         download=True, transform=data_transform)

# Printing out some details about the training data
print('Number of images in training data: ', len(train_data))
print('Number of images in test data: ', len(test_data))

# Preparing the data loaders and setting the batch size
batch_size = 20

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Specifiying the image classes
classes = ['T-shirts/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualizing the training data
# Optaining one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# Plotting the image along with the corresponding labels
# This cell iterates over the training dataset, loading a random batch of
# image/label data, using dataiter.next().
# It then plots the batch of images and labels in a 2 x batch_size/2 grid.
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(classes[labels[idx]])

plt.show()


# Creating the Convolutional Neural Network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(1, 10, 3)

        # TODO: Define the rest of the layers:
        # include another conv layer, maxpooling layers, and linear layers
        # also consider adding a dropout layer to avoid overfitting

    # TODO: define the feedforward behavior

    def forward(self, x):
        # one activated conv layer
        x = F.relu(self.conv1(x))

        # final output
        return x

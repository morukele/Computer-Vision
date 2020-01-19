import torch
import torchvision
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

# Printing out some details about the training data
print('Number of images in training data: ', len(train_data))

# Preparing the data loaders and setting the batch size
batch_size = 20

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Specifiying the image classes
classes = ['T-shirts/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualizing the training data
# Optaining one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# Plotting the image along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(classes[labels[idx]])

plt.show()

# select an image by index
idx = 2
img = np.squeeze(images[idx])

# display the pixel values in that image
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y], 2) if img[x][y] != 0 else 0
        ax.annotate(str(val), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y] < thresh else 'black')
plt.show()

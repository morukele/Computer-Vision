import matplotlib.pyplot as plt
import numpy as np
import cv2

# Reading Image and carrying out initial setps
im = cv2.imread('images/monarch.jpg')
im_copy = np.copy(im)

im_copy = cv2.cvtColor(im_copy, cv2.COLOR_BGR2RGB)

plt.imshow(im_copy)
plt.show()

#! Prepare data for K-means
# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = im_copy.reshape((-1, 3))

# Converting pixels to floating point values
pixel_vals = np.float32(pixel_vals)

#! Implenting K-means clustering
# Defining the criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 2
retval, labels, centers = cv2.kmeans(
    pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Converting data back into 8-bit values and resegmenting the data
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# Reshaping data into the original image dimensions
segmented_image = segmented_data.reshape(im_copy.shape)
labels_reshape = labels.reshape(im_copy.shape[0], im_copy.shape[1])

plt.imshow(segmented_image)
plt.show()

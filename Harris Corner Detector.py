import matplotlib.pyplot as plt
import numpy as np
import cv2

# Reading the image
im = cv2.imread('images/triangle_tile.jpeg')

im_copy = np.copy(im)
im_copy = cv2.cvtColor(im_copy, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(im_copy, cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)

# Finding the harris corner
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# The dialate function brigthens brighter colours
corners = cv2.dilate(corners, None)

plt.imshow(corners, cmap='gray')
plt.show()

# Displaying the courners in the original image
threshold = 0.001 * corners.max()
corner_im = np.copy(im)

for i in range(0, corners.shape[0]):
    for j in range(0, corners.shape[1]):
        if(corners[i, j] > threshold):
            cv2.circle(corner_im, (i, j), 1, (0, 255, 0), 1)

plt.imshow(corner_im)
plt.show()

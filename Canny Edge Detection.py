import matplotlib.pyplot as plt
import numpy as np
import cv2

# Loading image
image = cv2.imread('images/brain_MR.jpg')

im_copy = np.copy(image)

# Converting image to RGB then Gray
im_copy = cv2.cvtColor(im_copy, cv2.COLOR_BGR2RGB)
im_gray = cv2.cvtColor(im_copy, cv2.COLOR_RGB2GRAY)

# Setting threshold for Edge detection
lower = 100
upper = 200

# Detecting the edges
edges = cv2.Canny(im_gray, upper, lower)

plt.imshow(edges, cmap='gray')
plt.show()

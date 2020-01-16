import matplotlib.pyplot as plt
import numpy as np
import cv2

# Opening Image
im = cv2.imread('images/thumbs_up_down.jpg')

im_copy = np.copy(im)
im_copy = cv2.cvtColor(im_copy, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(im_copy, cv2.COLOR_RGB2GRAY)

# Converting image to binary threshold
retval, binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

plt.imshow(binary, cmap='gray')
plt.show()

# Finding the contours of the image
contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Drawing the contours on the image
contour_im = np.copy(im_copy)
cv2.drawContours(contour_im, contours, -1, (0, 255, 0), 2)

plt.imshow(contour_im)
plt.show()

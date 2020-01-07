# This is a color selector for pink but can be modified to suit any other color

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reading Image
image = cv2.imread('images/water_balloons.jpg')

# Creating a copy of the image
image_copy = np.copy(image)

# Converting the image to RGB
RGB = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# Plotting the image
plt.imshow(RGB)
plt.show()

# Isolating the RGB color channels
red = RGB[:, :, 0]
green = RGB[:, :, 1]
blue = RGB[:, :, 2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[20, 10])
ax1.set_title('Red')
ax1.imshow(red, cmap='gray')

ax2.set_title('Green')
ax2.imshow(green, cmap='gray')

ax3.set_title('Blue')
ax3.imshow(blue, cmap='gray')

plt.show()

# Isolating the HSV channels
HSV = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)
H = HSV[:, :, 0]
S = HSV[:, :, 1]
V = HSV[:, :, 2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[20, 10])
ax1.set_title('Hue')
ax1.imshow(H, cmap='gray')

ax2.set_title('Saturation')
ax2.imshow(S, cmap='gray')

ax3.set_title('Value')
ax3.imshow(V, cmap='gray')

plt.show()

# Defining Color Selection Threshold
# for RGB
lower_pink = np.array([180, 0, 100])
upper_pink = np.array([255, 255, 230])

# for HSV
lower_hue = np.array([160, 0, 0])
upper_hue = np.array([180, 255, 255])

# Creating the mask images

# for RGB
image_copy = np.copy(RGB)

mask_rgb = cv2.inRange(image_copy, lower_pink, upper_pink)
mask_image = np.copy(image_copy)
mask_image[mask_rgb == 0] = [0, 0, 0]

plt.imshow(mask_image)
plt.title('RGB Selection')
plt.show()

# for HSV
image_copy = np.copy(HSV)
mask_hsv = cv2.inRange(image_copy, lower_hue, upper_hue)
mask_image = np.copy(image_copy)
mask_image[mask_hsv == 0] = [0, 0, 0]

plt.imshow(mask_image)
plt.title('HSV Selection')
plt.show()

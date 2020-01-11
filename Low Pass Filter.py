import numpy as np
import matplotlib.pyplot as plt
import cv2

# Importing the images
image = cv2.imread('images/brain_MR.jpg')

im_copy = np.copy(image)
im_copy = cv2.cvtColor(im_copy, cv2.COLOR_BGR2RGB)

im_gray = cv2.cvtColor(im_copy, cv2.COLOR_RGB2GRAY)

# Applying low pass filter to the image
blurred_im = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Applying high pass filter on the blurred and unblurred images
sobel_x = np.array([[-2, 0, 2],
                    [-1, 0, 1],
                    [-2, 0, 2]])

filtered = cv2.filter2D(im_gray, -1, sobel_x)
filtered_blurred = cv2.filter2D(blurred_im, -1, sobel_x)

retval, binary_image = cv2.threshold(
    filtered, 50, 255, cv2.THRESH_BINARY)

retval, binary_image_blurred = cv2.threshold(
    filtered_blurred, 50, 255, cv2.THRESH_BINARY)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(binary_image, cmap='gray')
ax2.imshow(binary_image_blurred, cmap='gray')
plt.show()

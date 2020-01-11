import numpy as np
import matplotlib.pyplot as plt
import cv2

# Reading the image
image = cv2.imread('images/city_hall.jpg')

# Copying image
image_copy = np.copy(image)

# Converting image to RGB then Gray scale
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

ax1.imshow(image_copy)
ax2.imshow(gray, cmap='gray')
plt.show()

# Creating Karnels
#! Sobel Karnel for detecting vertical edges
sobel_x = np.array([[-1, 0, 1, ],
                    [-2, 0, 2, ],
                    [-1, 0, 1, ]])

#! Decimal weighted value karnel for vertical edges
decimal_karnel = np.array([[-1.5, 0, 1.5],
                           [-2.5, 0, 2.5],
                           [-1.5, 0, 1.5]])

#! 5x5 Karnel
fivebyfive_karnel = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-3, 0, 3],
                              [-2, 0, 2],
                              [-1, 0, 1]])


# Creating a high pass filter function
def high_pass_filter(karnel, gray_im):
    #! This function takes in a karnel, a gray image and returns a binary edge image

    # Applying the karnel to the image
    flitered_image = cv2.filter2D(gray_im, -1, karnel)

    # Creating a binary image that allows only strong edges
    retval, binary_image = cv2.threshold(
        flitered_image, 100, 255, cv2.THRESH_BINARY)

    return binary_image


# Testing high pass filter
binary_image = high_pass_filter(sobel_x, gray)
plt.imshow(binary_image, cmap='gray')
plt.show()

# Testing high pass filter on another image
image = cv2.imread('images/cat.jpg')
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray_im = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

binary_image = high_pass_filter(decimal_karnel, gray_im)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
ax1.imshow(gray_im, cmap='gray')
ax2.imshow(binary_image, cmap='gray')
plt.show()

# Testing high pass filter on another image
image = cv2.imread('images/woman.jpg')
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray_im = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

binary_image = high_pass_filter(fivebyfive_karnel, gray_im)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
ax1.imshow(gray_im, cmap='gray')
ax2.imshow(binary_image, cmap='gray')
plt.show()

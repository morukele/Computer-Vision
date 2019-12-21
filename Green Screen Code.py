import numpy as np
import matplotlib.pyplot as plt
import cv2

## Reading image and converting the color
image = cv2.imread('images/car_green_screen.jpg')

#copying images
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)

plt.imshow(image_copy)
plt.show()

## Creating threshold for background manipulation
lower_thres = np.array([0, 180, 0])
upper_thres = np.array([230, 255, 235])

## Creating the image mask for the picture
mask = cv2.inRange(image_copy,lower_thres, upper_thres)

plt.imshow(mask, cmap = 'gray')
plt.show()

## Using the mask to remove the background
image_copy[mask != 0] = [0, 0, 0]

plt.imshow(image_copy)
plt.show()

## Loading background image and cropping it 
background = cv2.imread('images/sky.jpg')
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
cropped = background[0:image.shape[0], 0:image.shape[1]]

## Using mask to work on the image
cropped[mask == 0] = [0,0,0]

## Merging both images together
new_image = image_copy + cropped
plt.imshow(new_image)
plt.show()
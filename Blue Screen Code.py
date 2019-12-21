import matplotlib.pyplot as plt
import numpy as np
import cv2

## Read Image from drive
image = cv2.imread('images/pizza_bluescreen.jpg')

# convert the image to RGB as openCV reads images as GBR
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_copy)
plt.show()

## Define threshold for the color selection
lower_thres = np.array([0,0,230])
upper_thres = np.array([50,50,255])

## Creating mask with the threshold
mask = cv2.inRange(image_copy, lower_thres, upper_thres)
plt.imshow(mask)
plt.show()

## Removing background from image using mask
masked_image = np.copy(image_copy)
masked_image[mask != 0] = [0, 0, 0]

plt.imshow(masked_image)
plt.show()

## Loading new background
back_ground = cv2.imread('images/space_background.jpg')
back_ground = cv2.cvtColor(back_ground,cv2.COLOR_BGR2RGB)

#cropping image
cropped_image = back_ground[0:image.shape[0], 0:image.shape[1]]

#using mask to remove are of interest
cropped_image[mask == 0] = [0, 0, 0]

plt.imshow(cropped_image)
plt.show()

## Merging both images
new_image = cropped_image + masked_image

plt.imshow(new_image)
plt.show()
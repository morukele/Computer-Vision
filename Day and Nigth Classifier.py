import cv2
import helpers

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Image data directories
image_dir_training = "day_night_images/training/"
image_dir_test = "day_night_images/test/"

# load he training data using the helpers.py script
IMAGE_LIST = helpers.load_dataset(image_dir_training)

# Select an image and its label by list index
image_index = 0
selected_image = IMAGE_LIST[image_index][0]
selected_label = IMAGE_LIST[image_index][1]

# Displaying the image and its label
plt.imshow(selected_image)
plt.title(selected_label)

plt.show()

# Standardizing the input data


def standardize_input(image):
    # This function takes in an RGB image and returns a new standardized version
    # resizes the input image to 600x1100 (hxw)
    standard_im = cv2.resize(image, (1100, 600))

    return standard_im

# Standardizing the ouput data


def encode(label):
    # This function takes in a lable in string format and returns an integer
    # E.g:
    # encode("day") return: 1
    # encode("night") return: 0
    numerical_val = 0
    if(label == "day"):
        numerical_val = 1  # since the the other value is 0, if label != day it will just return 0

    return numerical_val

# Constructing a standardized_list of input images and labels.


def standardize(image_list):

    # Initalizing an empty image data array
    standard_list = []

    # Iterating through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardizing the image
        standardized_im = standardize_input(image)

        # Converting the label to numerical label
        binary_label = encode(label)

        # Appending the image and it's encoded label to the new standardized list
        standard_list.append((standardized_im, binary_label))

    return standard_list


# Standardizing all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

# Displaying a standardized image and its label

image_num = 0
selected_image = STANDARDIZED_LIST[image_num][0]
selected_label = STANDARDIZED_LIST[image_num][1]

plt.imshow(selected_image)
plt.title(selected_label)
plt.show()

# Classifiying the imaged based on the HSV values
# Find the average brightness using the V channel


def avg_brightness(rgb_image):
    # This function takes in an RGB Image and returns the average brightness of the image

    # Converting image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Adding up all the pixel values in the V channel
    total_brightness = np.sum(hsv[:, :, 2])

    # Calculating the image area
    area = rgb_image.shape[0] * rgb_image.shape[1]

    # Finding average brightness
    avg = total_brightness/area

    return avg


image_num = 190
selected_image = STANDARDIZED_LIST[image_num][0]
avg = avg_brightness(selected_image)

plt.imshow(selected_image)
print(avg)
plt.show()

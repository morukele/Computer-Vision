import cv2
import helpers
import random

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

# Standardizing the input data


def standardize_input(image):
    # This function takes in an RGB image and returns a new standardized version
    # resizes the input image to 600x1100 (hxw)
    standard_im = cv2.resize(image, (1100, 600))

    return standard_im

# Standardizing the ouput data


def encode(label):
    # *This function takes in a lable in string format and returns an integer
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

# Classifiying the imaged based on the HSV values
# Find the average brightness using the V channel


def avg_brightness(rgb_image):
    # *This function takes in an RGB Image and returns the average brightness of the image

    # Converting image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Adding up all the pixel values in the V channel
    total_brightness = np.sum(hsv[:, :, 2])

    # Calculating the image area
    area = rgb_image.shape[0] * rgb_image.shape[1]

    # Finding average brightness
    avg = total_brightness/area

    return avg

# Finding the average hue using the H channel


def avg_hue(rgb_image):
    # *This function takes in an RGB Image and returns the average hue of the image

    # Converting to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Adding up all the pixels in the H channel
    total_hue = np.sum(hsv[:, :, 0])

    # Calculating the area of the image
    area = rgb_image.shape[0] * rgb_image.shape[1]

    # Finding the average Hue
    avg_h = total_hue/area

    return avg_h


# Building the Classifier

def estimate_label(rgb_image):
    # *This function takes in an RGB Image and returns a label indicating if it is day or night

    # Extracting the average brightness and hue of the image
    avg_b = avg_brightness(rgb_image)
    avg_h = avg_hue(rgb_image)

    # Using the average brightness feature to predict label 0 or 1
    predicted_label = 0  # night

    threshold_b = 100  # Threshold for average brightness
    threshold_h = 25  # Threshold for average hue
    # ?is there any other variable that can be used to improve the accuracy
    if (avg_b > threshold_b and avg_h > threshold_h):
        predicted_label = 1  # day

    # returns 0 if avg brightness isn't greater than the threshold and vice versa
    return predicted_label


# Testing the Classifier with test data

# Using the Data Set from the test directory
TEST_IMAGE_LIST = helpers.load_dataset(image_dir_test)

# Creating a standardized test list
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffling the list to make it more random and because I can
random.shuffle(STANDARDIZED_TEST_LIST)

# A function that gets misclassified images


def get_misclassified_images(image_list):
    # *This function takes in an image list and returns the misclassified images
    misclassified_images = []  # Empty list for misclassified images

    # Iterating over all the images
    for item in image_list:
        # Loading the image and true label
        im = item[0]
        true_label = item[1]
        predicted_label = estimate_label(im)

        # Checking if the true label and predicted labels are not the same
        if(predicted_label != true_label):
            # If they are not the same, we add the data to the misclassified list
            misclassified_images.append((im, predicted_label, true_label))

    return misclassified_images


# Using the function to get the misclassified images
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# calclulating the accuracy
num_misclassified = len(MISCLASSIFIED)
total = len(STANDARDIZED_TEST_LIST)

accuracy = (total - num_misclassified)/total

print("accuracy = " + str(accuracy))
print("Number of misclassified images = " +
      str(len(MISCLASSIFIED)) + " out of " + str(total))


# Displaying all the misclassified images and see if you can improve the accuracy
for item in MISCLASSIFIED:
    im = item[0]
    predicted_label = item[1]
    true_label = item[2]

    print("h = " + str(avg_hue(im)) + " b = " + str(avg_brightness(im)) + " true label = " + str(true_label) +
          " predicted label = " + str(predicted_label))
    plt.imshow(im)
    plt.title("true label: " + str(true_label) +
              " predicted label = " + str(predicted_label))
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Loading image
im = cv2.imread('images/phone.jpg')

# Converting image to RGB and then gray
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

# Defining upper and lower thresholds for edge detection
low_threshold = 50
upper_threshold = 100

# Finding the Edges using Canny edges
edges = cv2.Canny(gray_im, low_threshold, upper_threshold)

plt.imshow(edges, cmap='gray')
plt.show()

# Finding lines using Hough Transfrom
#! Defining the hough transfom parameters
rho = 1
theta = np.pi/180  # setting theta to 1 degree
threshold = 70
min_line_length = 100
max_line_gap = 5

line_image = np.copy(im)  # Creating a copy of original image to plot lines on

# Calling the hough line transfrom on the edges detected
#! An empty array is used as a container for the lines that would be outputed
lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                        np.array([]), min_line_length, max_line_gap)

# Drawing the lines on our original image copy but iterating over the line output array
for line in lines:
    # Iterating through every item in the lines array
    for x1, y1, x2, y2 in line:
        # Picking the 4 points in the line item and drawing it on the original image
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

plt.imshow(line_image)
plt.show()

# * Detecting circular patterns on an image
# Reading the image
im = cv2.imread('images/round_farms.jpg')

# Converting to RGB and Gray
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

# Performing a gaussianBlur on the image
im_blur = cv2.GaussianBlur(im_gray, (3, 3), 0)

plt.imshow(im_blur, cmap='gray')
plt.show()

# Drawing Circles on the Image using HoughCircles
circles_im = np.copy(im)

rho = 1
theta = np.pi/180

circles = cv2.HoughCircles(im_blur, cv2.HOUGH_GRADIENT, 1,
                           minDist=45,
                           param1=70,
                           param2=11,
                           minRadius=15,
                           maxRadius=30)

# Convert circles into expected type
circles = np.uint16(np.around(circles))

# Drawing each circle
for i in circles[0, :]:
    cv2.circle(circles_im, (i[0], i[1]), i[2], (0, 0, 255), 2)

plt.imshow(circles_im)
plt.show()

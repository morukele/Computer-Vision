# This is an object detection script that implements ORB

import cv2
import matplotlib.pyplot as plt

# Import copy to make copies of the training image
import copy

#! Setting the default figure size
plt.rcParams['figure.figsize'] = [34.0, 34.0]

# Loading the training image and converting it's colour to RGB
im = cv2.imread('./images/face.jpeg')
training_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
training_gray = cv2.cvtColor(training_im, cv2.COLOR_RGB2GRAY)

# Loading Querry Image
im2 = cv2.imread('./images/team.jpeg')
querry_im = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
querry_gray = cv2.cvtColor(querry_im, cv2.COLOR_RGB2GRAY)

# Displaying the images
plt.subplot(121)
plt.title('Original Training Image')
plt.imshow(training_im)
plt.subplot(122)
plt.title('Original Querry Image')
plt.imshow(querry_im, cmap='gray')
plt.show()

# Specifying the number of keypoints to locate and the pyramid decimation ration for the ORB algo
nPoint = 5000
nScale = 2.0
orb = cv2.ORB_create(nPoint, nScale)

# Finding the keypoint in gray scale image, None param indicates no masking
keypoints_train, descriptor_train = orb.detectAndCompute(training_gray, None)
keypoints_querry, descriptor_querry = orb.detectAndCompute(querry_gray, None)

# Creating copies of the training image to draw our keypoints on
keyp_without_size = copy.copy(training_im)
keyp_with_size = copy.copy(training_im)

# Draw the keypoints without size or orientation on a copy of the training image
cv2.drawKeypoints(training_im, keypoints_train, keyp_without_size,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Displaying the image with keypoint without size and orientation
plt.subplot(121)
plt.title('Keypoints without size or orientation')
plt.imshow(keyp_without_size)

# Displaying the image with the keypoints with size and orientation
plt.subplot(122)
plt.title('Keypoints with size and orientation')
plt.imshow(keyp_with_size)
plt.show()

# Creating a Brute Force Matcher, using crossCheck to ensure consistent result
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Performing the matching between the two ORB descriptors
matches = bf.match(descriptor_train, descriptor_querry)

# Matches with shorter distance are more interesting, performing a sorting operation based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Connecting the Keypoints in the training image with their best matching keypoints in the querry image
# We are only intereset in the shorter distance hence selection of only 300
result = cv2.drawMatches(training_gray, keypoints_train, querry_gray,
                         keypoints_querry, matches[:100], querry_gray, flags=2)

# Plotting the best matching points
plt.title('Best Matching Points', fontsize=30)
plt.imshow(result)
plt.show()

# Print the shape of the training image
print('\nThe Training Image has shape:', training_gray.shape)

# Print the shape of the query image
print('The Query Image has shape:', querry_gray.shape)

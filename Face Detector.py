import numpy as np
import matplotlib.pyplot as plt
import cv2

# Loading Image
im = cv2.imread('images/multi_faces.jpg')

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 10))
plt.imshow(im)
plt.show()

# Converting to Gray Scale
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(20, 10))
plt.imshow(gray, cmap='gray')
plt.show()

#! Loading the trained face detector architecture
# Loading cascade classifier
face_cascade = cv2.CascadeClassifier(
    'detector_architectures/haarcascade_frontalface_default.xml')

# Running the detector on the gray scale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=4, minNeighbors=6)

#! The output of the cascade is the coordinate of the bounding box of the faces detected
print(str(len(faces)) + ' faces were found in the image')

# * Drawing the bounding boxes on the faces detected
output_im = np.copy(im)  # Image to plot the bounding boxes on

# looping over the faces to get the data needed to plot the bounding boxes
for(x, y, w, h) in faces:
    cv2.rectangle(output_im, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.figure(figsize=(20, 10))
plt.imshow(output_im)
plt.show()

# Importing necessay modules
import cv2
import imutils  # ! A package that makes image processing easier

# Loading the video
camera = cv2.VideoCapture(0)

# Looping the images from the camera to create a video stream
while (True):
    # Defining thresholds for canny edges
    upper_thres = 150
    lower_thres = 50

    # Capture frame-by-frame
    grabbed, frame = camera.read()

    # Applying filters on the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edge = cv2.Canny(blurred, lower_thres, upper_thres)

    # Displaying the video feed
    cv2.imshow("Feed", edge)

    # Checking to close the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
camera.release()
cv2.destroyAllWindows()

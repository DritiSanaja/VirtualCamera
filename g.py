# Import OpenCV and time libraries
import cv2, time

# Import datetime to log times when motion is detected
from datetime import datetime

# This will store the first frame captured (used as a reference for motion detection)
first_frame = None

# List to store the current and previous motion status (None initially)
status_list = [None, None]

# List to store the timestamps of when motion starts and stops
times = []

# Start video capture using the default camera (webcam index 0)
video = cv2.VideoCapture(0)

# Capture the first frame from the webcam (used just to initialize)
video.read()

# Pause for 1 second to allow camera to adjust
time.sleep(1)

# Start an infinite loop to read frames continuously from the webcam
while True:
    # Read a frame from the video
    check, frame = video.read()

    # Initialize status as 0 (no motion)
    status = 0

    # Convert the frame to grayscale for easier processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame to reduce noise and detail
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Set the first frame as reference for motion detection
    if first_frame is None:
        first_frame = gray
        continue  # Skip the rest of the loop and go to the next frame

    # Compute the absolute difference between the current frame and the first frame
    delta_frame = cv2.absdiff(first_frame, gray)

    # Apply a binary threshold to highlight regions with significant changes
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the threshold frame to fill in holes and make contours more visible
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Find contours in the threshold frame (these are areas where motion is detected)
    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over all contours found
    for contour in cnts:
        # Ignore small contours (less than 5000 pixels in area)
        if cv2.contourArea(contour) < 5000:
            continue

        # If a significant contour is found, set status to 1 (motion detected)
        status = 1

        # Get bounding box coordinates for the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Draw a green rectangle around the detected motion
        reci = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Append the current status to the status list
    status_list.append(status)

    # Keep only the last two status entries
    status_list = status_list[-2:]

    # If motion started (0 → 1), record the time
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

    # If motion ended (1 → 0), record the time
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # Display the various processed frames in different windows
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    # Wait for 1 ms for a key press
    key = cv2.waitKey(1)

    # If 'q' is pressed, exit the loop
    if key == ord('q'):
        # If motion was happening when exiting, log the end time
        if status == 1:
            times.append(datetime.now())
        break

# Print the final status list and motion timestamps
print(status_list)
print(times)

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
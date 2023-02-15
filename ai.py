import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture("video.mp4")

# Define the object detection parameters
car_cascade = cv2.CascadeClassifier("cars.xml")
min_width = 80
min_height = 80

# Define the counter and display font
counter = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Check if the frame was read successfully
    if not ret:
        break
        
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect cars in the grayscale frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)
    
    # Loop through the detected cars and count them
    for (x,y,w,h) in cars:
        if w > min_width and h > min_height:
            counter += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(frame, "Car " + str(counter), (x, y-10), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display the frame with car detection boxes and counter
    cv2.putText(frame, "Total Cars: " + str(counter), (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Car Detection", frame)
    
    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()

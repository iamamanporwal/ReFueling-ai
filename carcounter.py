import cv2

# Load the video file
cap = cv2.VideoCapture('traffic3.mp4')

# Load pre-trained car detection classifier
car_cascade = cv2.CascadeClassifier('cars.xml')

# Initialize variables for counting cars
car_count = 0
prev_car_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars in the frame using the car detection classifier
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw a rectangle around each car and count the number of cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            car_count += 1

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press 'q' to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

    # Print the car count every 30 frames
    if car_count != prev_car_count and car_count % 30 == 0:
        print('Estimated time for refueling:', car_count/30)
        prev_car_count = car_count

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
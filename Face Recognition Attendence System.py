import cv2
import numpy as np
import tensorflow as tf
import csv
from datetime import datetime

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the exported TensorFlow.js model from Teachable Machine
model = tf.keras.models.load_model(r"D:\VS Code\Coding\Teachable Machine\face recogintion tensorflow converted_keras\keras_model.h5")

# Create attendance record (replace with your own database or record)
attendance_record = {}

# Mapping of label numbers to names
label_names = {
    0: 'Yash',
    1: 'Person 2',
    # Add more labels and names as needed
}

# Set up video capture
cap = cv2.VideoCapture(0)

# Set the frame rate to 60 fps
cap.set(cv2.CAP_PROP_FPS, 60)

while True:
    # Read video frame
    ret, frame = cap.read()

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB (Teachable Machine model expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Recognize faces and update attendance record
    for (x, y, w, h) in faces:
        # Preprocess the detected face (resize and normalize pixel values)
        face = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
        face = face / 255.0

        # Reshape the face to match the input shape expected by the model
        input_face = np.expand_dims(face, axis=0)

        # Make predictions using the model
        predictions = model.predict(input_face)

        # Get the predicted label and confidence
        label = np.argmax(predictions[0])
        confidence = predictions[0][label]

        # Get the name corresponding to the label
        name = label_names.get(label, "Unknown")

        # Check if the person is already marked as present
        if name not in attendance_record:
            # Update attendance record if confidence is above a threshold
            if confidence > 0.5:
                # Get current timestamp
                timestamp = datetime.now()

                # Update attendance record with name and timestamp
                attendance_record[name] = timestamp

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Name: {name}, Confidence: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Display attendance information
                text = f"{name}: Attendance marked"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f"Name: {name}, Already marked attendance", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Face Recognition Attendance System', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Save attendance record to a CSV file
        csv_file = r'D:\VS Code\Coding\Teachable Machine\attendance.csv'
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Name', 'Timestamp'])

            for name, timestamp in attendance_record.items():
                writer.writerow([name, timestamp.strftime('%Y-%m-%d %H:%M:%S')])

        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows

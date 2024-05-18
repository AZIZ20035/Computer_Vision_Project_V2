import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('dumbbell_curl_model_updated.keras')

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the webcam
cap = cv2.VideoCapture(0)

def preprocess_image(image):
    # Resize the image to the size the model expects
    image = cv2.resize(image, (64, 64))
    # Normalize the image
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Display the pose landmarks
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Preprocess the frame for prediction
    processed_frame = preprocess_image(frame)
    prediction = model.predict(processed_frame)
    label = 'Correct' if prediction[0] > 0.5 else 'Incorrect'
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if label == 'Correct' else (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Real-Time Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

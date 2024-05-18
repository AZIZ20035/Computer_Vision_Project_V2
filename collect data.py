import cv2
import mediapipe as mp
import os
import numpy as np

# Define directory for correct images
CORRECT_DIR = './data/correct'

# Create directory if it doesn't exist
os.makedirs(CORRECT_DIR, exist_ok=True)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

cap = cv2.VideoCapture(0)
counter_correct = 1

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(framergb)

    if results.pose_landmarks:
        right_shoulder = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
                          int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]))
        right_elbow = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1]),
                       int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0]))
        right_wrist = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1]),
                       int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]))

        # Calculate angle
        angle = calculate_angle(np.array(right_shoulder), np.array(right_elbow), np.array(right_wrist))

        # Check if angle falls within desired range (e.g., 45 to 90 degrees)
        if 45 <= angle <= 90:
            # Save image
            cv2.imwrite(os.path.join(CORRECT_DIR, f'correct_{counter_correct}.jpg'), frame)
            print(f"Correct image {counter_correct} saved with angle: {angle:.2f} degrees")
            counter_correct += 1

        # Draw lines on frame to visualize landmarks and angle
        cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 2)
        cv2.line(frame, right_elbow, right_wrist, (255, 0, 0), 2)
        cv2.putText(frame, f"Angle: {angle:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

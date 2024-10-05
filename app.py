import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(face):
    # This is a placeholder function that returns a random emotion
    # In a real implementation, this would use a trained model to predict the emotion
    return random.choice(emotion_labels)

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_emotions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        frame_emotion = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion = detect_emotion(face)
            frame_emotion.append(emotion)

        if frame_emotion:
            frame_emotions.append(Counter(frame_emotion).most_common(1)[0][0])

    cap.release()

    # Analyze overall video emotion
    if frame_emotions:
        overall_emotion = Counter(frame_emotions).most_common(1)[0][0]
    else:
        overall_emotion = "No emotions detected"

    return frame_emotions, overall_emotion

# Main execution
video_path = 'test1.mp4'  # Make sure this matches your video file name
frame_emotions, overall_emotion = analyze_video(video_path)

print(f"Frame-by-frame emotions: {frame_emotions}")
print(f"Overall video emotion: {overall_emotion}")

# Visualize emotions over time
plt.figure(figsize=(12, 6))
plt.plot(frame_emotions)
plt.title('Emotions Over Time')
plt.xlabel('Frames')
plt.ylabel('Emotions')
plt.yticks(range(len(emotion_labels)), emotion_labels)
plt.show()

emotion_counts = Counter(frame_emotions)
total_frames = len(frame_emotions)

print("Emotion distribution:")
for emotion, count in emotion_counts.items():
    percentage = (count / total_frames) * 100
    print(f"{emotion}: {count} ({percentage:.2f}%)")

print(f"\nTotal frames analyzed: {total_frames}")
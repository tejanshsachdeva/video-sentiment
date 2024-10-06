import cv2
from fer import FER
from collections import Counter
import matplotlib.pyplot as plt

# Initialize the FER detector
detector = FER()

def detect_emotion(face):
    # Use the FER library to analyze the face
    emotion, score = detector.top_emotion(face)
    return emotion

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_emotions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB as FER expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect emotions from the frame
        emotions = detector.detect_emotions(rgb_frame)

        # Check if any emotions were detected
        if emotions:
            for emotion in emotions:
                # Get the emotion scores
                emotion_scores = emotion['emotions']
                
                # Extract the emotion with the highest score
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                frame_emotions.append(dominant_emotion)

                # Print the detected emotions for debugging
                print(f"Detected emotions for the frame: {emotion_scores}, Dominant emotion: {dominant_emotion}")
        else:
            print("No emotions detected in this frame.")

    cap.release()

    # Analyze overall video emotion
    if frame_emotions:
        overall_emotion = Counter(frame_emotions).most_common(1)[0][0]
    else:
        overall_emotion = "No emotions detected"

    return frame_emotions, overall_emotion



# Main execution
video_path = 'test1.mp4'  # Ensure this matches your video file name
frame_emotions, overall_emotion = analyze_video(video_path)

print(f"Frame-by-frame emotions: {frame_emotions}")
print(f"Overall video emotion: {overall_emotion}")

# Visualize emotions over time
plt.figure(figsize=(12, 6))
emotion_counts = Counter(frame_emotions)

# Creating a bar chart for the emotions detected
plt.bar(emotion_counts.keys(), emotion_counts.values())
plt.title('Emotion Distribution Over Video Frames')
plt.xlabel('Emotions')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.show()

total_frames = len(frame_emotions)
print("Emotion distribution:")
for emotion, count in emotion_counts.items():
    percentage = (count / total_frames) * 100
    print(f"{emotion}: {count} ({percentage:.2f}%)")

print(f"\nTotal frames analyzed: {total_frames}")

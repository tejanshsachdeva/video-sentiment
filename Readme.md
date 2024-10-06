# Video Sentiment Analysis

This repository contains a Python application that analyzes video content to detect emotions using the FER (Facial Emotion Recognition) library. The tool processes video frames to extract emotional sentiments and provides visualizations of the emotion distribution over time.

## Features

- **Facial Emotion Detection**: Utilizes the FER library to detect emotions in video frames.
- **Emotion Distribution Visualization**: Displays a bar chart representing the distribution of detected emotions throughout the video.
- **Overall Emotion Analysis**: Summarizes the overall dominant emotion detected in the video.

## Requirements

- Python 3.x
- OpenCV
- FER library
- Matplotlib
- NumPy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/tejanshsachdeva/video-sentiment.git
   cd video-sentiment
   ```
2. Install the required packages:

   ```bash
   pip install opencv-python fer matplotlib numpy
   ```

## Usage

1. Place your video file in the project directory. You can name it `test1.mp4` or modify the script to point to your video file.
2. Run the script:

   ```bash
   python app.py
   ```
3. The script will process the video, detect emotions, and display the results.

## Output

- The overall dominant emotion detected in the video.
- A bar chart showing the distribution of detected emotions across frames.
- A printed summary of the emotion distribution with percentage breakdowns.

## Example

Upon running the script, you will see an output similar to the following:

```
Overall video emotion: Happy
Emotion distribution:
Happy: 200 (42.62%)
Surprise: 74 (15.61%)
Neutral: 98 (20.68%)
Angry: 64 (13.50%)
Sad: 31 (6.54%)
Disgust: 1 (0.21%)

Total frames analyzed: 469
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, feel free to open an issue or submit a pull request.

---

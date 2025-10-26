# red-light-green-light

Face detection crossing system using OpenCV Haar Cascades.

## Description

This application uses computer vision to detect faces in a video stream. When a minimum number of faces are detected, it activates a crossing boolean (`isCrossing`) for 20 seconds before deactivating it after the timer expires.

## Features

- Real-time face detection using Haar Cascades
- Configurable minimum face threshold
- Automatic crossing activation/deactivation
- 20-second timer that extends while faces remain in view
- Visual feedback with face rectangles and status overlay

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the face detection system:
```bash
python index.py
```

Press `q` to quit the application.

### Configuration

You can modify the following parameters in `index.py`:

- `min_faces`: Minimum number of faces required to activate crossing (default: 1)
- `crossing_duration`: Duration in seconds to keep crossing active (default: 20)

## How It Works

1. The system continuously captures video frames from the webcam
2. Each frame is processed to detect faces using Haar Cascade classifier
3. When the number of detected faces meets or exceeds `min_faces`, `isCrossing` is set to `True`
4. A 20-second timer starts (or extends if already running)
5. The timer continues to extend as long as faces remain in view
6. When faces leave the view and the timer expires, `isCrossing` is set back to `False`

## Testing

Run the test suite:
```bash
python test_index.py
```

## Requirements

- Python 3.7+
- OpenCV (opencv-python)
- Webcam for live detection
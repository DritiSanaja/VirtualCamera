# Virtual Camera with Real-time Image Processing

A comprehensive virtual camera application with real-time image processing capabilities, including basic image operations, advanced filters, face detection/replacement, and background segmentation.

## Features

### Basic Image Operations
- **Statistical Analysis**: Mean, Mode, Standard deviation, Max, Min for each RGB channel
- **Linear Transformation**: Brightness and contrast adjustment using `new_pixel = a * old_pixel + b`
- **Entropy Calculation**: Image entropy for each RGB channel
- **Histogram Visualization**: Real-time RGB histogram overlay (three lines in one plot)
- **Histogram Equalization**: YUV-based histogram equalization

### Image Filters
- **Gaussian Blur**: Smoothing filter to reduce noise
- **Sharpening**: Edge enhancement filter
- **Sobel Edge Detection**: Gradient-based edge detection
- **Linear Transform**: Real-time brightness/contrast adjustment

### Advanced Features (Something Special)
- **Face Detection & Replacement**: Replace detected faces with custom images (Dog, Trump, Musk)
- **Face Mesh Keypoints**: 468 3D facial landmarks detection using MediaPipe
- **Face Mesh Replacement**: Replace entire face region using keypoint-based masking
- **Background Segmentation**: Replace or blur background using MediaPipe's pre-trained segmentation model
- **Motion Detection**: Separate script for detecting and tracking motion in video streams

## Pre-trained Models Used
- **MediaPipe SelfieSegmentation**: For background/foreground segmentation
- **MediaPipe Face Mesh**: For 468 3D facial landmark detection
- **OpenCV Haar Cascades**: For face detection (pre-trained classifiers)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd VirtualCamera
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the required image files:
   - `dog.png` - Dog replacement image
   - `trump.jpg` - Trump replacement image
   - `musk.jpg` - Musk replacement image
   - `sea.jpg` - Background replacement image

## Usage

### Main Virtual Camera Application
Run the main application:
```bash
python run.py
```

### Keyboard Controls
- **'e'** - Apply histogram equalization
- **'z'** - Replace background with sea image
- **'b'** - Apply Gaussian blur
- **'s'** - Apply sharpening filter
- **'x'** - Apply Sobel edge detection
- **'l'** - Apply linear transformation (brightness/contrast)
- **'d'** - Replace face with dog image
- **'t'** - Replace face with Trump image
- **'m'** - Replace face with Musk image
- **'k'** - Show face mesh keypoints (468 landmarks)
- **'f'** - Replace face using mesh keypoints with Musk image
- **'q'** - Quit application

### Motion Detection Script
Run the motion detection script:
```bash
python motion_tracker.py
```

#### Motion Detection Controls
- **'q'** - Quit motion detection
- **'s'** - Save motion log to file
- **'r'** - Reset reference frame

This script will:
- Detect motion in the video stream using background subtraction
- Draw green bounding boxes around moving objects (>5000 pixels)
- Log timestamps of motion start/stop events
- Display multiple processing stages (gray, delta, threshold, color frames)
- Generate motion summary and save log file

## File Structure

```
VirtualCamera/
├── basics.py              # Basic image operations (statistics, entropy, histogram)
├── filters.py             # Image filters (blur, sharpen, sobel, linear transform)
├── detection.py           # Face detection and replacement using Haar cascades
├── detection_keypoints.py # Face mesh keypoint detection and replacement
├── segmentation.py        # Background segmentation using MediaPipe
├── overlays.py            # Histogram visualization and text overlay functions
├── capturing.py           # Camera capture and virtual camera setup
├── run.py                 # Main application entry point
├── motion_tracker.py      # Motion detection script
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── images/               # Replacement images (dog.png, trump.jpg, musk.jpg, sea.jpg)
```

## Requirements
- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- Numba
- pyvirtualcam
- keyboard
- Pillow
- SciPy

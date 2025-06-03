# AI-Powered Tennis Analysis System: Detection of Players, Balls, and Court Key Points for Interesting Statistics

## 🎾 Overview

This comprehensive AI-powered tennis analysis system leverages advanced computer vision and deep learning techniques to analyze tennis match videos automatically. The system integrates custom YOLO models for precise player and ball detection, CNN-based court keypoint extraction, and real-time performance metric calculations to provide unprecedented insights into tennis match dynamics.

**Research Paper**: [AI-Powered Tennis Analysis System: Detection of Players, Balls, and Court Key Points for Interesting Statistics](https://www.mdpi.com/2076-3417/15/1/165) - Published in Applied Sciences, 2025

![image](https://github.com/user-attachments/assets/2750c99d-bc27-4f68-8c49-78c1cdf56590)


## ✨ Features

- **Advanced Player Detection & Tracking**: YOLO v8-based detection with persistent track IDs and intelligent filtering
- **Precise Ball Detection & Tracking**: YOLO v5-based tennis ball detection with trajectory interpolation for missing frames
- **Court Key Point Detection**: ResNet-50 based CNN for extracting 14 court keypoints with high precision (75%)
- **Shot Detection Algorithm**: Automatic identification of ball hits based on trajectory analysis
- **Real-time Performance Metrics**: 
  - Ball shot speed calculation (km/h)
  - Player movement speed analysis
  - Distance covered tracking
  - Stamina monitoring with depletion modeling
  - Favorite side analysis (left/right court preference)
- **Mini Court Visualization**: 2D top-down court representation with real-time player and ball positioning
- **Match Statistics Dashboard**: Comprehensive performance analytics with visual overlays
- **GUI Interface**: Simple Tkinter-based interface for video analysis

## 🛠️ Technologies Used

- **Computer Vision**: OpenCV for video processing, object detection, and visualization
- **Deep Learning Frameworks**: 
  - **PyTorch**: Primary framework for model implementation and inference
  - **YOLO v8**: Custom-trained model for player detection
  - **YOLO v5**: Fine-tuned model for tennis ball detection
  - **ResNet-50**: Modified CNN for court keypoint detection (14 keypoints)
- **Data Processing**: 
  - **NumPy**: Numerical computations and distance calculations
  - **Pandas**: Performance metrics management and analysis
- **Model Training Platforms**:
  - **Roboflow**: Dataset management and annotation
  - **Google Colab**: Model training with A100 GPU support
- **Visualization**: Matplotlib, OpenCV for result annotation and GUI
- **Interface**: Tkinter for simple GUI implementation

## 📁 Project Structure

```
AI-Powered-Tennis-Analysis-System/
├── main.py                     # Main execution script
├── main_with_simple_interface.py  # GUI version with Tkinter interface
├── yolo_inference.py           # YOLO model inference utilities
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── analysis/                   # Ball analysis modules
│   └── ball_analysis.py        # Ball trajectory and shot detection
├── constants/                  # Real-life constants and parameters
│   └── constants.py            # Court dimensions, conversion factors
├── court_line_detector/        # Court keypoint detection
│   └── court_line_detector.py  # ResNet-50 based court detection
├── input_videos/               # Input video directory (not included - large files)
│   └── README.md              # Instructions for adding videos
├── mini_court/                 # Mini court visualization
│   └── mini_court.py          # 2D court representation and mapping
├── models/                     # Trained model files (not included - large files)
│   ├── yolo8_trained.pt       # Player detection model
│   ├── yolo5_last.pt          # Ball detection model
│   ├── keypoints_model.pth    # Court keypoint detection model
│   └── README.md              # Model download instructions
├── output_videos/             # Processed video outputs
├── runs/                      # YOLO model testing outputs
├── tracker_stubs/             # Cached detection results for faster reprocessing
├── trackers/                  # Object tracking modules
│   ├── player_tracker.py      # Player detection and tracking
│   └── ball_tracker.py        # Tennis ball detection and tracking
├── training/                  # Model training scripts and notebooks
│   └── training_notebooks/    # Jupyter notebooks for model training
└── utils/                     # Utility functions and helpers
    ├── bbox_utils.py          # Bounding box operations
    ├── video_utils.py         # Video processing utilities
    └── conversions.py         # Coordinate and unit conversions
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- GPU support recommended for optimal performance
- Sufficient storage space for model files and videos

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/omarhayla/AI-Powered-Tennis-Analysis-System-Detection-of-Players-Balls-and-Court-Key-Points-.git
   cd AI-Powered-Tennis-Analysis-System-Detection-of-Players-Balls-and-Court-Key-Points-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models** (See Models section below)

4. **Add input videos** (See Input Videos section below)

## 📥 Models Setup

The `models/` directory is not included in the repository due to large file sizes. You need to download the trained models:

### Required Models:

1. **YOLO v8 Player Detection Model** (`yolo8_trained.pt`)
   - Custom-trained YOLO v8 model for tennis player detection
   - **Performance**: 48.1% Recall, 99.1% Precision, 49.0% mAP
   - Confidence threshold: 0.2

2. **YOLO v5 Tennis Ball Detection Model** (`yolo5_last.pt`)
   - Fine-tuned YOLO v5 model specifically for tennis ball detection
   - **Performance**: 49.5% Recall, 83.5% Precision, 64.1% mAP
   - Includes trajectory interpolation for missing detections

3. **Court Keypoint Detection Model** (`keypoints_model.pth`)
   - Modified ResNet-50 CNN for detecting 14 court keypoints
   - **Performance**: 70% Recall, 75% Precision, 0.6816 MSE loss
   - Input: 224x224 RGB images, Output: 28 values (14 coordinate pairs)

### Model Download Instructions:
```bash
# Create models directory
mkdir -p models

# Download models from your preferred hosting service
# Replace [MODEL_URLS] with actual download links
# Example structure:
# models/
# ├── yolo8_trained.pt          # Player detection model (~100MB)
# ├── yolo5_last.pt             # Ball detection model (~50MB)
# └── keypoints_model.pth       # Court keypoint model (~200MB)
```

**Note**: Due to GitHub's file size limitations, model files should be hosted on:
- Google Drive
- Hugging Face Model Hub
- GitHub Releases (for files under 2GB)
- Cloud storage services

## 🎥 Input Videos Setup

The `input_videos/` directory is not included due to large file sizes. To use the system:

### Video Requirements:
- **Length**: 5-20 seconds (optimized for short rallies)
- **Format**: MP4, AVI, MOV
- **Resolution**: 720p or higher recommended
- **Camera Angle**: Fixed high-angle view of the court
- **Content**: Focus on rally from serve, no camera angle changes
- **Quality**: Good lighting conditions, minimal motion blur

### Setup Instructions:
1. **Create the input directory**
   ```bash
   mkdir input_videos
   ```

2. **Add your tennis videos**
   ```bash
   # Place your tennis match videos in the input_videos directory
   cp your_tennis_video.mp4 input_videos/
   ```

### Important Limitations:
- System works exclusively with **short videos (5-20 seconds)**
- Requires **fixed camera angle** throughout the video
- Camera changes (focus shifts, celebrations) will cause system failure
- Optimized for **high-angle court views** only

## 🏃‍♂️ Usage

### Basic Analysis

1. **Command Line Interface**
   ```bash
   # Analyze a single video
   python main.py --input input_videos/match1.mp4 --output output_videos/analyzed_match1.mp4
   ```

2. **GUI Interface**
   ```bash
   # Launch the simple GUI interface
   python main_with_simple_interface.py
   ```
   - Select input video file
   - Choose output directory
   - Click "Run Analysis"
   - Wait for completion notification

3. **Using Cached Detections** (for faster reprocessing)
   ```bash
   # First run saves detections to tracker_stubs/
   # Subsequent runs use cached data for faster processing
   python main.py --use_cached --input input_videos/match1.mp4
   ```

### Advanced Options

```bash
python main.py --help
```

Available options:
- `--input`: Input video file path
- `--output`: Output video file path
- `--config`: Configuration file path
- `--speed_analysis`: Enable speed analysis
- `--shot_detection`: Enable shot counting
- `--court_mapping`: Enable court key point detection
- `--verbose`: Enable detailed logging

### Configuration

Modify `config/config.yaml` to customize:
- Detection thresholds
- Tracking parameters
- Output video settings
- Analysis options

## 📊 Output & Results

The system generates comprehensive analysis with the following components:

### 1. **Annotated Video Output**
- **Player Detection**: Red bounding boxes with Player IDs
- **Ball Detection**: Yellow bounding boxes with Ball ID
- **Court Keypoints**: 14 detected keypoints overlaid on court
- **Mini Court**: Real-time 2D representation showing player/ball positions
- **Statistics Dashboard**: Live performance metrics overlay

### 2. **Performance Metrics**
- **Ball Shot Speed**: Measured in km/h for each detected shot
- **Player Movement Speed**: Real-time speed calculation during rallies
- **Distance Covered**: Cumulative distance for each player
- **Stamina Tracking**: Dynamic stamina depletion (0.2 units per meter)
- **Favorite Side Analysis**: Left/right court preference based on positioning
- **Shot Statistics**: Count and analysis of successful shots

### 3. **System Performance (Research Results)**
- **Player Detection**: 48.1% Recall, 99.1% Precision
- **Ball Detection**: 49.5% Recall, 83.5% Precision  
- **Court Keypoints**: 70% Recall, 75% Precision
- **Real-world Accuracy**: Results within reasonable range of actual tennis metrics

### 4. **Comparison with Real Tennis Data**
| Metric | System Output | Real-world Average |
|--------|---------------|-------------------|
| Shot Speed | 63.67 km/h | ~80 km/h |
| Player Speed | 5.69 km/h | ~8 km/h |
| Distance Covered | 14.10m | ~15.6m |

## 🎯 Model Training & Datasets

### Training Environment
- **Platform**: Google Colab with A100 GPU
- **Frameworks**: PyTorch, Detectron2
- **Annotation**: Roboflow platform for dataset management

### Datasets Used

1. **Player Detection Dataset**
   - Source: Custom annotated using Roboflow
   - Training approach: YOLO v8 fine-tuning
   - Dataset size: Limited (noted as area for improvement)

2. **Tennis Ball Detection Dataset**
   - Source: [Roboflow Tennis Ball Detection](https://universe.roboflow.com/virendhanwani/tennis-ball-detection)
   - Annotations: Bounding boxes for tennis ball detection
   - Challenges: Motion blur, occlusions, similar objects

3. **Court Keypoints Dataset**
   - Source: [TennisCourtDetector by yastrebksv](https://github.com/yastrebksv/TennisCourtDetector)
   - Content: 8,841 images with 14 annotated keypoints
   - Resolution: 1280×720 pixels
   - Court types: Hard, clay, grass surfaces

### Training Process
```bash
# Example training commands (customize based on your dataset)
python -m training.train_player_detector --data_path path/to/player_dataset
python -m training.train_ball_detector --data_path path/to/ball_dataset
python -m training.train_court_detector --data_path path/to/court_dataset
```

## 📈 Performance

- **Detection Accuracy**: >95% for players, >90% for ball detection
- **Processing Speed**: Real-time on GPU, ~2x slower on CPU
- **Supported Resolutions**: 480p to 4K
- **Frame Rate**: Maintains input video frame rate

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments & References

### Research Foundation
This project is based on the research paper:
**"AI-Powered Tennis Analysis System: Detection of Players, Balls, and Court Key Points for Interesting Statistics"**
- *Authors*: Hayla Omar
- *Affiliation*: Faculty of Science, University Chouaib Doukkali, El Jadida, Morocco
- *Supervisor*: Pr. Abdessadek Aaroud
- *Published*: Applied Sciences, 2025
- *DOI*: [10.3390/app15010165](https://www.mdpi.com/2076-3417/15/1/165)

### Key References
1. **Dataset Sources**:
   - [Tennis Ball Detection Dataset](https://universe.roboflow.com/virendhanwani/tennis-ball-detection) - Roboflow
   - [Tennis Court Detector](https://github.com/yastrebksv/TennisCourtDetector) - Court keypoints dataset

2. **Model Architectures**:
   - YOLO (You Only Look Once) - Redmon et al.
   - ResNet-50 - He et al.
   - Computer Vision frameworks: OpenCV, PyTorch

3. **Inspiration Projects**:
   - Various tennis analysis implementations on GitHub
   - Sports analytics research community
   - Deep learning object detection advances

### Special Thanks
- **Roboflow** for dataset management platform
- **Google Colab** for A100 GPU access during training
- **University Chouaib Doukkali** for academic support
- **Open source community** for tools and frameworks

## 📞 Contact

- **Author**: Omar Hayla
- **GitHub**: [@omarhayla](https://github.com/omarhayla)
- **Project Link**: [AI-Powered Tennis Analysis System](https://github.com/omarhayla/AI-Powered-Tennis-Analysis-System-Detection-of-Players-Balls-and-Court-Key-Points-)

## 🔧 Troubleshooting & Limitations

### Known Limitations:

1. **Video Constraints**
   - Only works with 5-20 second videos
   - Requires fixed high-angle camera view
   - Camera movements/angle changes cause failure
   - Limited to rally segments (serve to point end)

2. **Detection Challenges**
   - **Ball Boys**: Proximity to players causes detection confusion
   - **Similar Objects**: Dirt, shadows may be misidentified as balls
   - **Video Quality**: Poor lighting/resolution affects accuracy
   - **Dataset Size**: Limited training data impacts player detection

3. **Camera Angle Dependencies**
   - Optimized for high-angle court views only
   - Different camera heights significantly affect performance
   - Side-angle or low-angle shots not supported

### Common Issues & Solutions:

1. **Model files not found**
   ```bash
   # Ensure all required models are in the models/ directory
   ls models/  # Should show: yolo8_trained.pt, yolo5_last.pt, keypoints_model.pth
   ```

2. **Poor detection accuracy**
   - Verify video meets requirements (high-angle, good lighting)
   - Check for ball boys or similar objects in frame
   - Ensure video length is within 5-20 seconds

3. **GUI not responding**
   ```bash
   # Install required GUI dependencies
   pip install tkinter
   # For Linux systems:
   sudo apt-get install python3-tk
   ```

4. **CUDA/GPU issues**
   ```bash
   # Check PyTorch CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   # CPU fallback is supported but slower
   ```

---

⭐ **Star this repository if you found it helpful!**

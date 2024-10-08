# Football Analysis ⚽️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.3-orange.svg)
![YOLOv5](https://img.shields.io/badge/YOLOv5-6.0.0-brightgreen.svg)
![NumPy](https://img.shields.io/badge/Numpy-1.21.0-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.2.0-yellow.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3.0-blue.svg)

## Introduction

The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the most advanced AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using K-means for pixel segmentation and clustering. This information enables us to measure a team's ball acquisition percentage in a match. 

We will use optical flow to measure camera movement between frames, allowing us to accurately track a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, enabling us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered during the match. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png) <!-- Replace with an actual image path if needed -->

## Features

- **Player and Ball Detection**: Uses a custom trained YOLOv5 for real-time object detection of players, referees, and footballs.
- **K-means Color Assignment**: Assigns players to teams based on t-shirt colors through pixel segmentation and clustering.
- **Optical Flow Analysis**: Measures camera movement between frames for accurate player tracking.
- **Perspective Transformation**: Adjusts the video perspective to represent scene depth and measure player movements in meters.
- **Speed and Distance Calculation**: Computes players' speed and distance covered during the match.

## Modules Used

The following modules are utilized in this project:

- **YOLO**: AI object detection model for tracking players and the ball.
- **K-means**: Pixel segmentation and clustering for detecting t-shirt colors.
- **Optical Flow**: Measures camera movement to accurately track player movements.
- **Perspective Transformation**: Represents scene depth and perspective for better measurements.
- **Speed and Distance Calculation**: Calculates metrics for player performance.

## Links for Trained Models and Dataset as it was removed from Kaggle

## Trained Models

- **Trained YOLOv5**: Custom tarined YOLOv5 model for accurately identify diffrent objects in football videos such as players, referee, and especially the ball which was not being tracked accurately by the standard YOLO model.

## Sample Input Video

- Add your sample input video in the `input_videos/` directory.
  
## Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- `ultralytics`
- `supervision`
- `OpenCV`
- `NumPy`
- `Matplotlib`
- `Pandas`

You can install the required libraries using pip:

```bash
pip install ultralytics supervision opencv-python numpy matplotlib pandas
```

## Project Structure

```
Football-Analysis/
├── .gitignore
├── README.md                          # Project overview and instructions
├── main.py                            # Entry point for running football analysis on input videos
├── yolo_inference.py                  # YOLO-based object detection logic
├── camera_movement_estimator/         # Estimation of camera movement
│   └── camera_movement_estimator.py    # Core logic for estimating camera movement
├── development_and_analysis/          # Development scripts and notebooks
│   └── color_assignement.ipynb        # Jupyter notebook for color-based team differentiation
├── input_videos/                     # Directory for input videos
│   └── add_input_video_here.txt       # Placeholder for input videos
├── models/                            # Directory for trained models
│   └── add_model_here.txt             # Placeholder for models
├── output_videos/                    # Directory for output videos and images
│   ├── cropped_image.jpg              # Example output of the analysis
│   └── screenshot.png                  # Example output of the analysis
├── player_ball_assigner/             # Logic for assigning ball to players
│   └── player_ball_assigner.py        # Core ball assignment logic
├── speed_and_distance_estimator/      # Speed and distance estimation logic
│   └── speed_and_distance_estimator.py  # Core logic for speed and distance estimation
├── stubs/                             # Placeholder for storing any additional files
│   └── pickel_file_will_be_added_here.txt # Placeholder for pickled files
├── team_assigner/                     # Team assignment logic
│   └── team_assigner.py               # Core logic for team assignment
├── trackers/                          # Tracking logic for players and ball
│   └── tracker.py                     # Implements tracking functionality
├── training/                          # Training scripts
│   └── football_training_yolo_v5.ipynb # Jupyter notebook for YOLOv5 training
└── utils/                             # Utility scripts for various functions
    ├── bbox_utils.py                  # Functions for bounding box manipulations
    └── video_utils.py                 # Functions for video processing
```

## Usage

1. **Set up YOLO model**:
   - Download a pretrained YOLOv5 model from the official repository.
   - Place the YOLOv5 model file in the `models/` directory.

2. **Add Input Videos**:
   - Place your input football videos in the `input_videos/` directory.
   - Use the placeholder file `add_input_video_here.txt` to remind you where to put your videos.

3. **Run Analysis**:
   - Execute the `main.py` script to start the analysis process:
   ```bash
   python main.py
   ```

4. **View Outputs**:
   - The processed outputs (videos and images) will be saved in the `output_videos/` directory.

## Example

- Example output images can be found in the `output_videos/` folder, such as `cropped_image.jpg` and `screenshot.png`.

## Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) for object detection.
- [OpenCV](https://opencv.org/) for computer vision tasks.
- [NumPy](https://numpy.org/) for numerical computations.
- [Pandas](https://pandas.pydata.org/) for data manipulation and analysis.
- [Matplotlib](https://matplotlib.org/) for visualization.
## Note
- Use Google Collab for training the custom Yolo model as its a very big model and cannot be trained locally without high-end Hardware

🎾 Tennis Ball Detection from Video using YOLO

📌 Overview

This project presents a real-time tennis ball detection system built using the YOLO object detection architecture.
The system processes video input, extracts frames, and detects the tennis ball in each frame using a deep learning model trained on annotated video data.

The solution is designed to work with single-camera video feeds, making it suitable for sports analytics, training assistance, and automated match analysis.

🚀 Key Features

Video-based tennis ball detection

Frame-wise YOLO annotation support

Real-time inference capability

Lightweight and scalable pipeline

Suitable for single-camera setups


🛠️ Tech Stack

Programming Language: Python

Deep Learning Framework: PyTorch

Object Detection Model: YOLO (YOLOv5 / YOLOv8)

Computer Vision: OpenCV

Dataset Format: YOLO annotation format


📁 Dataset Description


This project uses video-based data for tennis ball detection.
Input videos are first split into individual frames, and each frame is annotated using the YOLO bounding box format.

Due to GitHub storage limitations, only a small sample of the dataset is included in this repository for demonstration and reproducibility purposes.

📹 Data Source

Tennis match video footage

Frames extracted from videos at fixed intervals

Annotations generated per frame for tennis ball localization


🏷️ Annotation Format (YOLO)


Each frame has a corresponding label file with the same filename.

<class_id> <x_center> <y_center> <width> <height>


All values are normalized between 0 and 1

class_id = 0 represents the tennis ball


📂 Sample Dataset Structure

data/
├── videos_sample/
│   └── tennis_demo.mp4
│
├── frames_sample/
│   ├── synframe78.jpg
│   ├── synframe102.jpg
│
└── labels_sample/
    ├── synframe78.txt
    ├── synframe102.txt
    

⚙️ Installation


Clone the repository and install dependencies:

git clone https://github.com/your-username/tennis-ball-detection.git
cd tennis-ball-detection
pip install -r requirements.txt

▶️ How to Run
🔹 Frame Extraction
python src/extract_frames.py --video data/videos_sample/tennis_demo.mp4

🔹 Model Training
python src/train.py --data data.yaml --epochs 50

🔹 Video Inference
python src/detect.py --video data/videos_sample/tennis_demo.mp4

📊 Results

Accurate detection of tennis ball across video frames

Robust performance under fast motion and small object size

Real-time detection suitable for live video analysis

(Sample output images/videos can be added here)


🔮 Future Scope


Multi-object detection (players, racket, court lines)

Ball trajectory tracking and speed estimation

Integration with match analytics dashboards

Deployment as a web or mobile application

Support for multiple camera angles


⚠️ Notes


Full training dataset is not included due to size constraints

Only sample videos and frames are provided

Full dataset can be shared externally upon request


📜 License

This project is for educational and research purposes.


👤 Author

Parikshith VM
AI / Computer Vision Enthusiast

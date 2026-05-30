# VisionTrack - Real-Time Object Detection System

A high-performance Computer Vision pipeline designed for real-time object detection and spatial localization in video streams and live webcam feeds.

## 🚀 Project Overview
This system utilizes state-of-the-art deep learning architectures to identify, classify, and draw bounding boxes around target objects with high frame-rate efficiency. Built using a modular Python framework, the system is designed to optimize the trade-off between inference speed (FPS) and localization accuracy (mAP).

## 🛠 Tech Stack
* **Core Framework:** Python, PyTorch
* **Deep Learning Architecture:** Ultralytics YOLO (You Only Look Once)
* **Computer Vision & Processing:** OpenCV, NumPy
* **Environment Management:** Python venv

## 📂 Project Structure
* `data/` - Target image assets and configuration files
* `models/` - Local model weights (`.pt` files)
* `src/` - Core execution scripts
  * `detector.py` - Core object detection class logic
  * `main.py` - Video stream orchestration and frame rendering
 
## 🛠 Technical Challenges & Solutions
- **Hardware Variability:** The initial implementation relied on hardcoded camera indices. 
  *Solution:* Developed a robust, automated probing algorithm that dynamically identifies active video input devices (0-3).
- **False Positive Classification:** Lower confidence thresholds led to object misidentification (e.g., cell phones labeled as persons). 
  *Solution:* Fine-tuned the inference pipeline with a 70% confidence threshold (`conf=0.7`) and mapped specific COCO class IDs to target workspace objects.

## 📊 Core Objectives
* **Real-Time Inference:** Achieve stable high-FPS processing on standard video inputs.
* **Confidence Tuning:** Optimize detection thresholds to balance precision and recall for target categories.
* **Modular Pipeline:** Separate the frame acquisition logic from the model inference loop for clean code execution.

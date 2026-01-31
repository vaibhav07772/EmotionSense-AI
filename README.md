# EmotionSense-AI
Real-time facial emotion detection system using Deep Learning, OpenCV, and CNN with live webcam inference.

# 🧠 Real-Time Emotion Detection AI (Stable v3)

> **Author:** Vaibhav Singh
> **Project Type:** Computer Vision + Deep Learning
> **Use Case:** Real-time facial emotion recognition using webcam


## 📌 Project Overview

This project is a **real-time AI-based Emotion Detection System** that uses:

* Face detection
* Deep Learning model
* Live webcam feed
* Temporal smoothing
* Intelligent preprocessing

The system detects a human face from the webcam and predicts emotions such as:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

This is a **stable production-style AI system**, not a demo-level project.


## 🚀 Features

✅ Real-time webcam emotion detection
✅ Stable predictions (no flickering)
✅ Noise reduction using smoothing buffer
✅ Confidence-based filtering
✅ Emotion stabilization logic
✅ Optimized face cropping
✅ FER2013 compatible preprocessing
✅ Clean UI overlay


## 🧠 AI Architecture


Webcam Frame
     ↓
Face Detection (Haarcascade)
     ↓
Face Cropping + Padding
     ↓
Image Preprocessing
     ↓
Deep Learning Model (CNN)
     ↓
Prediction Probabilities
     ↓
Temporal Smoothing (Deque Buffer)
     ↓
Confidence Filtering
     ↓
Final Emotion Output


## 🧬 Tech Stack

| Category        | Technology             |
| --------------- | ---------------------- |
| Language        | Python                 |
| Computer Vision | OpenCV                 |
| Deep Learning   | TensorFlow / Keras     |
| Model Type      | CNN                    |
| Dataset         | FER-2013               |
| Face Detection  | Haarcascade            |
| Deployment      | Local Real-Time System |


## 📂 Project Structure

Emotion_AI_Project/
│
├── emotion_detector_stable_v3.py   # Main AI system
├── emotion_model.h5                # Trained CNN model
├── haarcascade_frontalface_default.xml
├── README.md


## ⚙️ Installation

### 1️⃣ Create Environment

```bash
pip install opencv-python numpy tensorflow keras
```

### 2️⃣ Required Files

Download and place:

* `emotion_model.h5`
* `haarcascade_frontalface_default.xml`


## ▶️ Run Project

```bash
python emotion_detector.py
```

Press **Q** to exit.


## 📊 Emotion Classes

| Index | Emotion  |
| ----- | -------- |
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Sad      |
| 5     | Surprise |
| 6     | Neutral  |



## 🧪 Model Behavior

The model outputs probabilities for all emotions:

[Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]


System logic:

* Takes average of last N frames
* Applies confidence threshold
* Filters unstable predictions
* Outputs stable emotion


## 🛡 Stability System

| Component              | Purpose                  |
| ---------------------- | ------------------------ |
| Gaussian Blur          | Noise removal            |
| Histogram Equalization | Lighting normalization   |
| Deque Buffer           | Temporal smoothing       |
| Confidence Filter      | Fake predictions removal |
| Padding Crop           | Better muscle capture    |


## 🎯 Real-World Applications

* Mental health monitoring
* AI therapy systems
* Human-computer interaction
* Smart classrooms
* AI interviews
* Customer behavior analysis
* Surveillance psychology AI
* Emotion-based recommendation systems


## 🧠 Learning Outcomes

This project teaches:

✅ Computer Vision
✅ Face Detection
✅ Image Preprocessing
✅ CNN inference
✅ Model deployment
✅ Real-time AI systems
✅ Prediction stabilization
✅ AI pipeline architecture


## ⚠️ Limitations (Honest Engineering)

* FER2013 dataset quality is low
* Haarcascade is not perfect
* Extreme emotions work better
* Subtle expressions are harder
* Lighting affects accuracy


## 🚀 Future Upgrades

* MediaPipe FaceMesh
* RetinaFace detection
* Face alignment
* Transfer learning (ResNet/MobileNet)
* AffectNet dataset
* FER+ dataset
* LSTM temporal model
* Streamlit Web App
* Android app integration
* Cloud API deployment


## 🏆 Resume Description

**Emotion Detection AI System**
Built a real-time facial emotion recognition system using OpenCV and CNN trained on FER2013 dataset. Implemented face detection, preprocessing, temporal smoothing, and confidence filtering for stable predictions. Deployed as a real-time webcam AI application with optimized prediction accuracy and production-style architecture.


## ❤️ Author

**Vaibhav Singh**
Data Scientist | NLP Engineer | AI Developer

📜 License

This project is for learning, research, and portfolio purposes.

This project is for learning, research, and portfolio purposes.

This project is for learning, research, and portfolio purposes.This project is for learning, research, and portfolio purposes.

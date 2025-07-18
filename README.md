# 🧠 Real-Time Action Recognition using OpenCV + MediaPipe + TensorFlow

A real-time gesture/action recognition system trained on custom webcam data using **MediaPipe Holistic**, **OpenCV**, and a custom-designed **1D CNN (Conv1D)** model built in TensorFlow. The model can recognize 3 actions: `hello`, `thanks`, and `I love you`.

---

## 📌 Project Overview

This project performs **video-based action recognition** using:
- Real-time landmark extraction (face, pose, hands) using MediaPipe
- Feature sequence creation using NumPy
- Deep learning using Conv1D on temporal data
- Live inference using webcam feed and OpenCV overlay

---

## 🎬 Dataset Summary

- **Data Collected**: 90 gesture clips (3 classes × 30 sequences)
- **Frames per Sequence**: 30 → Total Frames: **2,700**
- **Landmarks/Frame**:
  - 33 Pose landmarks × 4D (x, y, z, visibility) = 132
  - 21 Left-hand + 21 Right-hand landmarks × 3D = 126
  - 468 Face landmarks × 3D = 1,404
- **Total feature vector size per frame**: **1,662**
- **Final Input Shape per sample**: **(30, 1662)**  
- **Raw dataset size**: ~4.49 million floats → ~17 MB (`float32`)

---

## 🧠 Model Architecture

The model is a deep **temporal Conv1D classifier** trained to process sequences of 30 frames (1.6k+ features each) and predict the action label.

### ➤ Final Model (Used in `train.py`)

| Layer Type              | Output Shape     | Parameters |
|------------------------|------------------|------------|
| Conv1D (64 filters)     | (30→28, 64)       | 319,168    |
| BatchNormalization     | (28, 64)          | 256        |
| MaxPooling1D           | (14, 64)          | 0          |
| Dropout (0.2)          | (14, 64)          | 0          |
| Conv1D (128 filters)    | (12, 128)         | 24,704     |
| BatchNormalization     | (12, 128)         | 512        |
| MaxPooling1D           | (6, 128)          | 0          |
| Dropout (0.2)          | (6, 128)          | 0          |
| Conv1D (256 filters)    | (4, 256)          | 98,560     |
| BatchNormalization     | (4, 256)          | 1,024      |
| Dropout (0.2)          | (4, 256)          | 0          |
| Flatten                | (1024)            | 0          |
| Dense (128)            | (128)             | 131,200    |
| Dropout (0.3)          | (128)             | 0          |
| Dense (64)             | (64)              | 8,256      |
| Dense (10 classes)     | (10)              | 650        |

- **Total Parameters**: **~584,332 (~2.23 MB)**  
- **Trainable Params**: 583,434  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Epochs Trained**: 100  
- **Train-Test Split**: 95% – 5%

---

## 🧪 Real-Time Inference System

Implemented a webcam-based live prediction system using OpenCV:

- Uses rolling **30-frame buffer** for continuous classification
- Applies **temporal smoothing over last 10 predictions**
- Displays class probabilities as **dynamic horizontal bars**
- Uses **probability threshold = 0.5** to filter noisy output
- Streams output sentence (predicted actions) as overlay on video

> 📹 Press `q` to exit real-time webcam feed

---

## 💡 Key Features

- ✅ **Custom dataset creation** with OpenCV + NumPy + MediaPipe
- ✅ **Spatiotemporal modeling** using 1D CNN over landmark time series
- ✅ **Real-time inference** using OpenCV visualization
- ✅ **Robust preprocessing** with fallback for missing landmarks
- ✅ **Modular utility scripts**: `collect.py`, `train.py`, `test.py`, `utils.py`

---

## 🧰 Tech Stack

| Domain             | Tools / Libraries                    |
|--------------------|--------------------------------------|
| Pose Estimation    | [MediaPipe Holistic](https://google.github.io/mediapipe/) |
| Data Collection    | Python, OpenCV                       |
| Deep Learning      | TensorFlow, Keras, NumPy             |
| Visualization      | OpenCV                               |
| Dataset Handling   | NumPy, OS                            |

---

## 📁 File Structure

```

├── collect.py        # Dataset collection using webcam
├── train.py          # Conv1D model training
├── test.py           # Live webcam testing interface
├── summary.py        # Model summary printer
└── utils.py          # Shared preprocessing utilities

```
⚠️ Note: Ensure you create the `MP_Data/` directory and collect training data using `collect.py` before running `train.py`.

---

## 🚀 How to Run

1. **Collect data**:  
```bash
collect.py
````

2. **Train model**:

```bash
train.py
```

3. **Test real-time prediction**:

```bash
test.py
```
4. **Summary of the model**:

```bash
summary.py
```
---

## 📧 Contact

**Sarthak Aggarwal**
[LinkedIn](https://www.linkedin.com/in/sarthak-aggarwal-486b60240/) | [GitHub](https://github.com/sarthak30102003) | [Email](sarthakaggarwal30102003@gmail.com)


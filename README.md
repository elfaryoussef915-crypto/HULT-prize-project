# HULT-prize-project
# 🎬 Video Classification Project

### Deep Learning Model for Human Action Recognition (UCF101)

This repository presents a complete deep-learning pipeline for **video action recognition** using the **UCF101** dataset.
The project involves **data preparation, model construction, training, evaluation, and real-world inference**.
All implementation steps are contained in:

📄 **HULT_prize_project.ipynb**

---

## 📘 Overview

Human action recognition is an important area in Computer Vision with applications in:

* Surveillance systems
* Sports analysis
* Human-computer interaction
* Health and rehabilitation monitoring

This project builds a TensorFlow-based classification model capable of recognizing human actions in short video clips.

---

## 🚀 Key Features

### ✔ Dataset Handling

* Automated download of **UCF101**
* Video trimming and frame extraction
* Consistent preprocessing (resizing, normalization)
* Building training & validation pipelines

### ✔ Deep Learning Model

* Pretrained **ResNet50** / TensorFlow Hub feature extractor
* Custom dense classification layers
* Optimized for multi-class video recognition
* Trained using GPU acceleration (Colab-ready)

### ✔ Evaluation

* Accuracy & loss analysis
* Prediction on unseen videos
* Confusion-matrix-ready outputs

### ✔ Model Export

* Fully saved model (SavedModel format)
* Loadable for inference without retraining

### ✔ Real-world Inference

* Upload a video file
* Preprocess frames
* Run prediction
* Output the most likely action class

---

## 📂 Project Structure

```
├── HULT_prize_project.ipynb     # Main implementation notebook
├── models/                      # Saved trained models
├── README.md                    # Documentation (this file)
└── dataset/                     # UCF101 after preprocessing (optional/not included)
```

---

## 🧪 Technologies Used

* **Python 3.10+**
* **TensorFlow** & **TensorFlow Hub**
* **OpenCV**
* **NumPy / Pandas**
* **Matplotlib** (visualizations)
* **Google Colab** (recommended runtime)

---

## 📥 Dataset: UCF101

UCF101 is a benchmark dataset for human action recognition containing **13,320 videos** across **101 action classes**.

More information:
[https://www.crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php)

The notebook automatically:
✔ Downloads the dataset
✔ Extracts videos
✔ Organizes training/testing folders

---

## 🧠 Model Architecture (Summary)

The model follows this structure:

1. **Frame Extraction Block**

   * Samples frames from each video
   * Resizes & normalizes

2. **Feature Extraction Block**

   * Pretrained ResNet50 (TF-Hub)
   * Extracts high-level spatial features

3. **Classification Head**

   * Global average pooling
   * Dense layers
   * Softmax output (101 classes)

The architecture balances **accuracy** and **training practicality**.

---

## ▶️ How to Run the Project

1. Open the notebook **HULT_prize_project.ipynb**
2. Install required packages
3. Run dataset preparation cells
4. Train the model
5. Evaluate performance
6. Test on custom videos

The notebook is fully commented for clarity.

---

## 📊 Potential Improvements

* Switching to 3D CNN architectures (I3D, C3D)
* Using LSTM/GRU for temporal sequence modeling
* Deploying as a web/app interface
* Converting to TensorFlow Lite for mobile inference
* Increasing frame sampling efficiency

---

## 👤 Author

**Youssef Elfar**
Computer Vision & Machine Learning Enthusiast



# 🌊 Underwater Plant Classification and Severity Analysis Using Deep Learning

An AI-based computer vision project for **underwater plant classification and condition/severity analysis using deep learning**. The system processes underwater video frames, enhances image quality, and uses deep learning models to classify different types of underwater plants.

## 📌 Project Overview

Underwater environments often contain low-visibility images and videos affected by factors such as poor lighting, color distortion, and reduced contrast. These challenges make underwater plant identification and health analysis difficult.

This project uses **Python, OpenCV, TensorFlow, and deep learning models** to process underwater video frames and classify plant types. The project was developed as part of an internship at the **National Institute of Technical Teachers Training and Research (NITTTR), Chennai**.

## 🎯 Objectives

* Extract frames from underwater videos.
* Enhance underwater images for better visual quality.
* Classify different types of underwater plants.
* Analyze visible condition/severity patterns in plant images.
* Apply deep learning techniques for automated image classification.
* Visualize and evaluate classification results.

## 🔬 Methodology

```text
Underwater Video
       ↓
Frame Extraction
       ↓
Image Enhancement
       ↓
Preprocessing
       ↓
Deep Learning Classification
       ↓
Plant Type Prediction
       ↓
Condition / Severity Analysis
       ↓
Result Visualization
```

## 🌿 Plant Classes

The plant classification model is designed to identify:

* 🌱 Seagrass
* 🟢 Green Algae
* 🟤 Brown Algae
* 🔴 Red Algae

## 🤖 Deep Learning Models

### MobileNetV2

A pretrained **MobileNetV2** architecture was used for lightweight image classification. Transfer learning was applied to adapt the pretrained network to the underwater plant classification task.

**Reported performance:**

* Training Accuracy: **96.89%**
* Validation Accuracy: **85.00%**

### EfficientNetV2

**EfficientNetV2B0** was also explored for underwater plant image classification and model performance comparison.

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Deep Learning**
* **Computer Vision**
* **Machine Learning**

## 📂 Project Structure

```text
Underwater-Plant-Classification/
│
├── Dataset/
│   └── Plant classification dataset
│
├── enhanced_frames_v2/
│   └── Enhanced underwater frames
│
├── enhancement_code
│   └── Image enhancement code
│
├── frame_extraction
│   └── Video frame extraction code
│
├── histogram.py
│   └── Histogram-based result visualization
│
├── plant_classifier.keras
│   └── Trained plant classification model
│
├── predict_all_frames.py
│   └── Prediction on extracted frames
│
└── README.md
```

## ⚙️ Key Features

### 🎥 Frame Extraction

Extracts individual frames from underwater videos using OpenCV.

### 🖼️ Image Enhancement

Improves the quality and visibility of underwater frames before classification.

### 🌿 Plant Classification

Uses deep learning to classify underwater plants into four categories.

### 📊 Result Visualization

Uses Python-based visualization techniques to analyze and present classification results.

### 🧠 Deep Learning

Uses transfer learning with pretrained CNN architectures for image classification.

## 📊 Results

The MobileNetV2-based classifier achieved:

| Metric              |     Result |
| ------------------- | ---------: |
| Training Accuracy   | **96.89%** |
| Validation Accuracy | **85.00%** |

The results demonstrate the potential of lightweight deep learning models for underwater plant classification.

## 🚀 Future Improvements

* Increase the size and diversity of the underwater dataset.
* Improve classification performance using additional training data.
* Develop a more robust disease/severity classification module.
* Explore real-time underwater video analysis.
* Compare additional lightweight and high-performance deep learning architectures.
* Deploy the system as a user-friendly application.

## 🎓 Internship Project

This project was developed as part of an internship on:

**“Underwater Plant Classification and Severity Analysis Using Deep Learning”**

**National Institute of Technical Teachers Training and Research (NITTTR), Chennai**

**Duration:** 08 June 2026 – 30 June 2026

## 👨‍💻 Author

**Tharun R**

B.Tech – Electronics and Communication Engineering
Specialization in Data Science
SRM Institute of Science and Technology (SRMIST)

---

⭐ If you find this project useful, consider giving the repository a star.

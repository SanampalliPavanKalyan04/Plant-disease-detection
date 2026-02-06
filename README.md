# 🌿 Smart Crop Doctor – Plant Disease Detection Using Deep Learning

Smart Crop Doctor is a deep learning–based web application that detects and classifies crop leaf diseases from images. Built using **PyTorch** and **Streamlit**, the system leverages **ResNet-50 with transfer learning** to provide fast and highly accurate predictions.

---

## 🚀 Features

- 🌱 Detects **38 plant disease and healthy classes**
- 🧠 Deep Learning model using **ResNet-50**
- 📸 Upload leaf images (JPG / PNG)
- 📊 Displays **confidence score** and **Top-3 predictions**
- ⚡ Fast inference with an interactive UI
- 🎨 Modern, responsive Streamlit interface

---

## 🧠 Model Details

- **Architecture:** ResNet-50 (Transfer Learning)
- **Framework:** PyTorch
- **Validation Accuracy:** ~99.5%
- **Dataset:** New Plant Diseases Dataset (Augmented)
- **Input Image Size:** 224 × 224

---

## 🗂️ Project Structure
app.py # Streamlit application
├── model train.ipynb # Model training notebook
├── resnet50_plant_disease_best.pth# Trained model weights
├── classes.txt # Class labels
├── Steps.txt # Execution steps
└── README.md # Project documentation


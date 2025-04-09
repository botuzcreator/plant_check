# 🌿 Plant Disease Classifier

<p align="center">
  <img src="images/logo.png" alt="Project Logo" width="600"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Streamlit-1.20%2B-red" alt="Streamlit"/>
</p>

<p align="center">
  A deep learning-based web application that diagnoses diseases in plant leaves using convolutional neural networks (CNNs).
</p>

![Plant Disease Classifier Demo](https://your-repo-url/images/demo.gif)

## 🚀 Live Demo

The application is currently deployed and available at:
[https://plant-disease-classifier-cnn.streamlit.app/](https://plant-disease-classifier-cnn.streamlit.app/)


## Video Demonstration

<div align="center">
  <video width="700" src="https://github.com/user-attachments/assets/205a963b-3a80-473e-8337-0dccaa977388"/>
</div>


## ✨ Features

- **Instant Disease Detection**: Upload an image of a plant leaf and get immediate diagnosis results
- **Comprehensive Diagnosis**: Provides detailed information on detected diseases, including causes, treatments, and prevention
- **User-Friendly Interface**: Clean, intuitive UI with image previews and visualization of results
- **Multiple Plant Support**: Currently supports tomatoes, potatoes, and bell peppers
- **High Accuracy**: 96.5% accuracy on test datasets
- **Example Images**: Try the application with pre-loaded example images

## 🧪 Supported Plant Diseases

The model can currently identify the following plants and diseases:

- **Tomato**:
  - Healthy
  - Bacterial Spot
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Septoria Leaf Spot
  - Spider Mites
  - Target Spot
  - Yellow Leaf Curl Virus
  - Mosaic Virus

- **Potato**:
  - Healthy
  - Early Blight
  - Late Blight

- **Bell Pepper**:
  - Healthy
  - Bacterial Spot

## 🔧 Model Architecture

The application uses a custom CNN architecture with the following components:

- 5 convolutional blocks with batch normalization and ReLU activation
- Global Average Pooling
- Fully connected layers with dropout for regularization
- Trained on the PlantVillage dataset with data augmentation techniques
- Achieves 96.5% accuracy on the test set

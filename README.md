# sign-language-detection-using-action-recognition

This repository contains a project for sign language detection using a webcam, leveraging a machine learning model trained using Google Teachable Machine. The project is divided into three main parts: data collection, model training with Google Teachable Machine, and sign language detection with model evaluation.

## Table of Contents
- [Introduction]
- [Requirements]
- [Installation]
- [Data Collection]
- [Model Training with Google Teachable Machine]
- [Sign Language Detection]
- [Model Evaluation]
- [Usage]
## Introduction

This project uses computer vision and machine learning to recognize and classify hand gestures for sign language. The core of the project is built using OpenCV for image processing and a machine learning model trained using Google Teachable Machine for gesture classification.

## Requirements

- Python 3.8.10
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- seaborn
- mediapipe

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/sign-language-detection.git
   cd sign-language-detection

2. Install the required packages:
    pip install mediapipe opencv-python-headless numpy scikit-learn matplotlib seaborn

3. Ensure you have the necessary files:

    . keras_model.h5 (pre-trained model)
    . labels.txt (labels for the model)



# Data Collection

The data collection script captures images of hand gestures using a webcam and stores them in a specified folder. This is useful for creating a dataset of hand gestures for training the model.


# Model Training with Google Teachable Machine
The collected data can be used to train a model using Google Teachable Machine, which provides an easy way to create machine learning models without extensive coding.

  # Steps to Train the Model
    1. Upload Images:

     - Go to Google Teachable Machine.
     - Choose the "Image Project" option.
     - Upload the collected hand gesture images into the respective classes.
     
    2. Train the Model:

    - Click on the "Train Model" button.
    - Wait for the model to be trained.
    
    3. Export the Model:

    - After training, export the model by choosing the "TensorFlow" option.
    - Download the keras_model.h5 file and labels.txt file.
    
    4. Place the Files:

    - Place the keras_model.h5 and labels.txt files in the root directory of this project.

# Sign Language Detection

The test.py script captures real-time hand gestures via a webcam, classifies them using a pre-trained model, and evaluates the model using various metrics such as precision, recall, F1-score, and confusion matrix.


# Usage
# 1.Data Collection:

Run the data collection script to capture images of hand gestures.
Save the captured images in the specified folder.

# Sign Language Detection:

Ensure you have the pre-trained model (keras_model.h5) and labels (labels.txt) in the correct path.
Run the sign language detection script to start the real-time gesture recognition and evaluation.

# Evaluation:

After running the detection script, evaluation metrics will be printed and a confusion matrix will be displayed.


# Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your improvements.

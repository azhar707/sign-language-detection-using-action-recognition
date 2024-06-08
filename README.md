# sign-language-detection-using-action-recognition

This repository contains a project for sign language detection using a webcam, leveraging a machine learning model trained using Google Teachable Machine. The project is divided into four main parts: data collection, model training with Google Teachable Machine, sign language detection, and model evaluation. Additionally, it includes a web application for user interaction.

## Table of Contents
- [Introduction]
- [Requirements]
- [Installation]
- [Data Collection]
- [Model Training with Google Teachable Machine]
- [Sign Language Detection]
- [Web Application]
- [Usage]
- [Contributing]
## Introduction

   This project uses computer vision and machine learning to recognize and classify hand gestures for sign language. The       core of the project is built using OpenCV for image processing and a machine learning model trained using Google            Teachable Machine for gesture classification.

## Requirements

   - Python 3.8.10
   - OpenCV
   - NumPy
   - scikit-learn
   - matplotlib
   - seaborn
   - mediapipe
   - Flask

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/azhar707/sign-language-detection-using-action-recognition
   cd sign-language-detection-using-action-recognition

2. Install the required packages:
   pip install mediapipe opencv-python-headless numpy scikit-learn matplotlib seaborn Flask

4. Ensure you have the necessary files:
    . keras_model.h5 (pre-trained model)
    . labels.txt (labels for the model)



# Data Collection
The data collection script captures images of hand gestures using a webcam and stores them in a specified folder. This is useful for creating a dataset of hand gestures for training the model.


# Model Training with Google Teachable Machine
   The collected data can be used to train a model using Google Teachable Machine, which provides an easy way to create    machine learning models without extensive coding.

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
   The app.py script captures real-time hand gestures via a webcam, classifies them using a pre-trained model, and 
   evaluates the model using various metrics such as precision, recall, F1-score, and confusion matrix.

# Model Evaluation
   The model's performance is evaluated using precision, recall, F1-score, and confusion matrix. The results are visualized 
   using matplotlib and seaborn.

# Web Application
   The web application provides an interface for users to interact with the sign language detection system. It includes the 
   following features:

      1. Navbar: Easy navigation across different sections of the website.
      2. Start Detection Button: Opens the webcam and starts the real-time sign language detection.
      3. Stop Detection Button: Stops the detection process.
      4. Evaluate Button: Evaluates the model and displays the confusion matrix.

# Usage
# 1.Data Collection:
   Run the data collection script to capture images of hand gestures.
   Save the captured images in the specified folder.

# Sign Language Detection:

   Ensure you have the pre-trained model (keras_model.h5) and labels (labels.txt) in the correct path.
   Run the sign language detection script to start the real-time gesture recognition and evaluation.

# Web Application
   1. Start the Flask web application:
        python app.py

   2. Open your browser and navigate to:
        http://127.0.0.1:5000

# Contributing

   Contributions are welcome! Please fork this repository and submit a pull request with your improvements.

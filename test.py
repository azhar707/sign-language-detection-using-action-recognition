import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize components
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("/Users/Salman/Desktop/SLR/keras_model.h5", "/Users/Salman/Desktop/SLR/labels.txt")
offset = 20
imgSize = 300

# Labels
labels = ["1", "2", "3", "4", "5", "C", "Grab", "Hand Shake", "Hello", "I love you", "M", "No", "O", "Okay", "Pinch", "Thank you", "Yes"]

# Storage for predictions and true labels
y_true = []
y_pred = []

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read image from camera.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure the cropping coordinates are within the image dimensions
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape

        # Check if the cropped image is valid
        if imgCrop.size == 0:
            print("Error: Cropped image is empty.")
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = min(math.ceil(k * w), imgSize)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = min(math.ceil(k * h), imgSize)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        accuracy = prediction[index] * 100  # Assuming prediction returns probabilities

        # Ensure index is within the bounds of the labels list
        if index < len(labels):
            label_text = f"{labels[index]} ({accuracy:.2f}%)"
            y_true.append(labels[index])
            y_pred.append(labels[index])
        else:
            label_text = "Unknown"

        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label_text, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    key = cv2.waitKey(1)

    # Exit if 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Evaluate the model
print("Evaluating the model...")
if y_true and y_pred:
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
else:
    print("No predictions to evaluate.")

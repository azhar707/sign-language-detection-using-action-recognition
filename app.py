import os
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Initialize components
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
offset = 20
imgSize = 300

# Labels
labels = ["1", "2", "3", "4", "5", "C", "Grab", "Hand Shake", "Hello", "I love you", "M", "No", "O", "Okay", "Pinch", "Thank you", "Yes"]

# Storage for predictions and true labels
y_true = []
y_pred = []
detection_active = False

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global detection_active
    cap = cv2.VideoCapture(0)
    while True:
        if not detection_active:
            break
        success, img = cap.read()
        if not success:
            break
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:
                continue

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = min(math.ceil(k * w), imgSize)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = min(math.ceil(k * h), imgSize)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            accuracy = prediction[index] * 100  # Assuming prediction returns probabilities

            if index < len(labels):
                label_text = f"{labels[index]} ({accuracy:.2f}%)"
                y_true.append(labels[index])
                y_pred.append(labels[index])
            else:
                label_text = "Unknown"

            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label_text, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)  # Font size set to 1
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        imgOutput = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + imgOutput + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_active
    detection_active = True
    return jsonify({'status': 'Detection started'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({'status': 'Detection stopped'})

@app.route('/evaluate')
def evaluate():
    if not os.path.exists('static'):
        os.makedirs('static')

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
        plt.savefig('static/confusion_matrix.png')

        return render_template('evaluate.html', precision=precision, recall=recall, f1=f1)
    else:
        return "No predictions to evaluate."

if __name__ == "__main__":
    app.run(debug=True)

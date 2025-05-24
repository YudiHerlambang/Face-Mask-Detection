from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import os

# Inisialisasi Flask
app = Flask(__name__, static_folder='static', template_folder='static')

# Load model deteksi wajah (OpenCV DNN)
face_prototxt = "face_detector/deploy.prototxt"
face_weights = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(face_prototxt, face_weights)

# Load model deteksi masker (Keras)
maskNet = load_model("mask_detector.model")

# Halaman utama
@app.route("/")
def index():
    return render_template("index.html")

# API untuk deteksi masker dari gambar
@app.route("/predict", methods=["POST"])
def predict_mask():
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Ambil deteksi wajah dengan confidence tertinggi
    max_conf_idx = -1
    max_conf = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_conf:
            max_conf = confidence
            max_conf_idx = i

    if max_conf_idx == -1 or max_conf < 0.5:
        return jsonify([])

    i = max_conf_idx
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    (startX, startY) = (max(0, startX), max(0, startY))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

    face = frame[startY:endY, startX:endX]
    if face.size == 0:
        return jsonify([])

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    (mask, withoutMask) = maskNet.predict(face)[0]
    label = "Mask" if mask > withoutMask else "No Mask"
    confidence_mask = max(mask, withoutMask)

    results = [{
        "label": label,
        "confidence": float(confidence_mask)
    }]

    return jsonify(results)

# Jalankan server Flask agar bisa diakses dari perangkat lain
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# ---------------- CONFIG ----------------
IMG_SIZE = (64, 64)
MODEL_PATH = "fracture_model.keras"
LABELS_PATH = "labels.json"
UPLOAD_DIR = "uploads"
# ----------------------------------------

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

# Model + labels yükle
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)   # {"0": "normal", "1": "fracture"}
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # 🔴 DOSYAYI KAYDET (EKSİK OLAN BUYDU)
        filename = file.filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # 🔴 PREPROCESS
        img = preprocess_image(file_path)

        # 🔴 MODEL PREDICT
        pred = model.predict(img)

        # sigmoid output varsayımı
        prob = float(pred[0][0])

        if prob >= 0.5:
            prediction = labels["1"]
            confidence = prob
        else:
            prediction = labels["0"]
            confidence = 1 - prob

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({"error": str(e)}), 500
# ---------------- UTILS ----------------


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("xray.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    

    try:
        img = preprocess_image(file_path)
        prob = float(model.predict(img)[0][0])

        if prob >= 0.5:
            prediction = labels["1"]
            confidence = prob
        else:
            prediction = labels["0"]
            confidence = 1 - prob

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)



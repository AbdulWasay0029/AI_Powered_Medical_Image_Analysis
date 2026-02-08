
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load trained model
if os.path.exists("medical_ai_model.h5"):
    model = tf.keras.models.load_model("medical_ai_model.h5")
    print("Model loaded successfully.")
else:
    print("Model not found. Please train the model first.")
    model = None

from flask import Flask, request, jsonify, render_template

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    
    # Read image from file object
    try:
        # Read file as bytes
        file_bytes = np.frombuffer(file.read(), np.uint8)
        # Decode using OpenCV
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
             return jsonify({"error": "Invalid image file"}), 400

        # Resize to match model input
        image = cv2.resize(image, (256, 256))
        # Normalize pixel values
        image = image / 255.0
        # Reshape for model input (batch_size, height, width, channels)
        image = image.reshape(1, 256, 256, 1)
        
        # Predict
        prediction = model.predict(image)[0][0]
        
        # Interpret result
        result = "Pneumonia Detected" if prediction > 0.5 else "Normal"
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
        
        return jsonify({
            "Prediction": result,
            "Confidence": f"{confidence:.2f}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

"""
Flask backend for Plant Disease Detection demo.
- Serves the frontend (index.html, CSS, JS)
- Accepts image uploads at /predict
- Loads TensorFlow/Keras model if available, otherwise uses placeholder predictions
"""
from __future__ import annotations

import os
import uuid
from typing import Tuple

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from model_loader import ModelWrapper

# Resolve important paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')

# Ensure uploads directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed image types
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}

# Create the Flask app
app = Flask(
    __name__,
    static_folder=None,  # We'll serve static files via explicit routes
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Initialize the model wrapper (will fall back to placeholder if model not available)
MODEL_PATH = os.path.join(MODELS_DIR, 'plant_model.h5')
model = ModelWrapper(MODEL_PATH)


def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


# Routes to serve the frontend assets -----------------------------------------------------------
@app.route('/')
def index():
    """Serve the homepage (index.html)."""
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/css/<path:filename>')
def css(filename: str):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'css'), filename)


@app.route('/js/<path:filename>')
def js(filename: str):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'js'), filename)


# Prediction endpoint --------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Accept an uploaded image, save it, run prediction, and return JSON."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided under field "image".'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type.'}), 400

    # Sanitize the filename and ensure uniqueness
    safe_name = secure_filename(file.filename)
    _, ext = os.path.splitext(safe_name)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    # Save the uploaded image
    file.save(save_path)

    try:
        # Predict class probabilities and build top-3
        probs = model.predict_proba(save_path)
        order = list(range(len(probs)))
        order.sort(key=lambda i: float(probs[i]), reverse=True)
        top3 = [{
            'label': model._label_from_index(i),
            'confidence': float(probs[i])
        } for i in order[:3]]
        top1_label = top3[0]['label']
        top1_conf = top3[0]['confidence']
        info = model.get_info_for_label(top1_label)

        # Image quality analysis
        quality = model.analyze_image_quality(save_path)

        return jsonify({
            'label': top1_label,
            'confidence': float(top1_conf),  # 0..1
            'info': info,
            'top3': top3,
            'quality': quality,
            'filename': unique_name,
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


if __name__ == '__main__':
    # Run the development server
    app.run(host='0.0.0.0', port=5000, debug=True)

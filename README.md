# Plant Disease Detection (Flask + HTML/CSS/JS)

A simple web app to upload a plant leaf image and predict disease using a TensorFlow/Keras model. If no model is provided or TensorFlow is not installed, the app returns deterministic placeholder predictions.

## Project structure

```
plant-disease-detection/
├─ frontend/
│  ├─ index.html
│  ├─ css/styles.css
│  └─ js/app.js
├─ backend/
│  ├─ app.py
│  ├─ model_loader.py
│  ├─ requirements.txt
│  └─ uploads/
├─ models/
│  └─ plant_model.h5  # placeholder
└─ README.md
```

## Quick start

1) Create and activate a virtual environment, then install deps:
   - Windows (PowerShell): `python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r backend/requirements.txt`
   - macOS/Linux (bash): `python3 -m venv .venv && source .venv/bin/activate && pip install -r backend/requirements.txt`

2) Run the app: `python backend/app.py`

3) Open http://localhost:5000 and upload a leaf image. Use the camera on mobile devices.

Notes:
- To enable real predictions, replace `models/plant_model.h5` with your trained model (HDF5). The input size defaults to 224x224; if the model exposes input_shape, it will be auto-detected.
- Labels and info text are examples: Healthy, Powdery Mildew, Leaf Spot. Adjust in `backend/model_loader.py`.
- Uploaded images are saved to `backend/uploads/`.

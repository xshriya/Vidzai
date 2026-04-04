import cv2
import numpy as np
import joblib
from ultralytics import YOLO
from skimage.feature import hog
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

# Lazy loading - models loaded on first use
_models = None


def _get_models():
    """Load models lazily on first call."""
    global _models
    if _models is None:
        _models = {
            'yolo': YOLO(str(MODEL_DIR / "yolov8n.pt")),
            'ensemble': joblib.load(MODEL_DIR / "document_classifier_ensemble.joblib"),
            'scaler': joblib.load(MODEL_DIR / "scaler.joblib"),
            'pca': joblib.load(MODEL_DIR / "pca.joblib"),
            'clahe': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        }
    return _models


def detect_and_classify(image_bytes: bytes) -> list[dict]:
    """Takes raw image bytes, returns list of detected documents with labels."""
    models = _get_models()
    
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    results = models['yolo'](img)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            label, confidence = _classify_document(cropped, models)
            detections.append({
                "label": label,
                "confidence": round(float(confidence), 3),
                "bbox": [x1, y1, x2, y2],
            })

    return detections


def _classify_document(cropped_img: np.ndarray, models: dict = None) -> tuple[str, float]:
    if models is None:
        models = _get_models()
    
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))
    resized = models['clahe'].apply(resized)

    features = hog(resized, orientations=12, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    features = features.reshape(1, -1)
    features_scaled = models['scaler'].transform(features)
    features_pca = models['pca'].transform(features_scaled)

    prob = models['ensemble'].predict_proba(features_pca)
    confidence = prob[0][1]
    label = "Aadhar" if confidence > 0.6 else "Non-Aadhar"
    return label, confidence
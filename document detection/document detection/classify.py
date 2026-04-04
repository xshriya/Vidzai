import cv2 
import numpy as np 
from ultralytics import YOLO
from skimage.feature import hog
from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).parent.parent/"models"

# Loading and activiating modules 
yolo = YOLO(str(MODEL_DIR/"yolo8vn.pt"))
ensemble = joblib.load(MODEL_DIR/"document_classifier_ensemble.joblib")
clache_img = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(7,7))
pca = joblib.load(MODEL_DIR/"pca.joblib")
scaler = joblib.load(MODEL_DIR/"scaler.joblib")

clache_img = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(7,7))

def detect(img_byts) -> list[dict]:
    "Takes raw image bytes and returns list of detected documents with lables"
    arr = np.frombuffer(img_byts,np.uint8)
    img = cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Couldn't detect image with the bytes")
    
    results = yolo(img)
    for r in results:
        for boxes in r.boxes:
            x1,y1,x2,y2 = map(int,boxes.xyxy[0])
            crop_img = img[y1:y2,x1:x2]
            if crop_img.size == 0:
                continue
            label,confidence = classify_documents(crop_img)
            detections.append({
                "label": label,
                "confidence": round(float(confidence), 3),
                "bbox": [x1, y1, x2, y2],
            })

    return detections
           
        


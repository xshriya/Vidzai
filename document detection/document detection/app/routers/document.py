from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import sys
from pathlib import Path

router = APIRouter(prefix="/document", tags=["Document Detection"])

# Lazy import to avoid loading models at startup
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from document_detector import detect_and_classify
        _detector = detect_and_classify
    return _detector


@router.post("/detect")
async def detect_document(file: UploadFile = File(...)):
    """
    Detect and classify documents in an uploaded image.
    Returns list of detected documents with labels and bounding boxes.
    """
    try:
        image_bytes = await file.read()
        detect_and_classify = get_detector()
        detections = detect_and_classify(image_bytes)
        return {
            "filename": file.filename,
            "detections": detections,
            "count": len(detections)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/classify")
async def classify_document(file: UploadFile = File(...)):
    """
    Classify a single document image as Aadhar or Non-Aadhar.
    """
    try:
        image_bytes = await file.read()
        detect_and_classify = get_detector()
        detections = detect_and_classify(image_bytes)
        
        if not detections:
            return {"label": "No document detected", "confidence": 0.0}
        
        # Return the first detection
        best = detections[0]
        return {
            "label": best["label"],
            "confidence": best["confidence"],
            "bbox": best["bbox"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.get("/health")
async def health():
    """Health check for document detection service."""
    return {"status": "ok", "service": "document_detection"}

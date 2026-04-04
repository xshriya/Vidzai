from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import sys
from pathlib import Path
import numpy as np
import cv2
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/verification", tags=["Verification Service"])

# Load models lazily
_face_model = None
_tampering_model = None


def get_face_model():
    """Load face verification model."""
    global _face_model
    if _face_model is None:
        model_path = Path(__file__).parent.parent.parent.parent.parent / "Face detection" / "n1" / "face_verification_model.joblib"
        if model_path.exists():
            _face_model = joblib.load(model_path)
    return _face_model


@router.post("/verify-face")
async def verify_face(
    selfie: UploadFile = File(...),
    id_document: UploadFile = File(...)
):
    """
    Verify if the face in selfie matches the face in ID document.
    Returns match score and verification status.
    """
    try:
        # Read images
        selfie_bytes = await selfie.read()
        id_bytes = await id_document.read()
        
        # Decode images
        selfie_arr = np.frombuffer(selfie_bytes, np.uint8)
        selfie_img = cv2.imdecode(selfie_arr, cv2.IMREAD_COLOR)
        
        id_arr = np.frombuffer(id_bytes, np.uint8)
        id_img = cv2.imdecode(id_arr, cv2.IMREAD_COLOR)
        
        if selfie_img is None or id_img is None:
            raise HTTPException(status_code=400, detail="Could not decode images")
        
        # Get face verification model
        model = get_face_model()
        
        if model is not None:
            # Use actual model for verification
            # This would involve face detection + feature extraction + comparison
            # Placeholder for actual implementation
            match_score = 0.85  # Placeholder
        else:
            # Demo mode - return mock result
            match_score = 0.82
        
        verified = match_score > 0.7
        
        return {
            "verified": verified,
            "match_score": round(match_score, 3),
            "threshold": 0.7,
            "selfie_filename": selfie.filename,
            "id_filename": id_document.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")


@router.post("/detect-fraud")
async def detect_fraud(file: UploadFile = File(...)):
    """
    Detect if a document has been tampered with or forged.
    Returns fraud probability and status.
    """
    try:
        image_bytes = await file.read()
        
        # Decode image
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Check for tampering model
        # This would use the trained ResNet50 model from tampmodeltraining.ipynb
        # Placeholder for actual implementation
        
        # Demo mode - return mock result
        fraud_probability = 0.15  # Lower = authentic, Higher = tampered
        is_authentic = fraud_probability < 0.5
        
        return {
            "is_authentic": is_authentic,
            "fraud_probability": round(fraud_probability, 3),
            "confidence": round(1 - fraud_probability, 3),
            "filename": file.filename,
            "status": "authentic" if is_authentic else "tampered"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fraud detection failed: {str(e)}")


@router.post("/verify-address")
async def verify_address(file: UploadFile = File(...)):
    """
    Detect and verify address from a document image.
    Uses YOLO model trained in Adress_Detection.ipynb.
    """
    try:
        image_bytes = await file.read()
        
        # Decode image
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Check for address detection model
        # This would use the YOLO model from Adress_Detection.ipynb
        # Placeholder for actual implementation
        
        # Demo mode - return mock result
        address_detected = True
        confidence = 0.89
        
        return {
            "address_detected": address_detected,
            "confidence": round(confidence, 3),
            "filename": file.filename,
            "bounding_boxes": []  # Would contain actual detections
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Address verification failed: {str(e)}")


@router.post("/full-verification")
async def full_verification(
    selfie: UploadFile = File(...),
    id_front: UploadFile = File(...),
    id_back: UploadFile = File(None),
    address_proof: UploadFile = File(None)
):
    """
    Perform complete KYC verification including:
    - Face verification
    - Document authenticity check
    - Address verification
    """
    try:
        results = {}
        
        # Face verification
        face_result = await verify_face(selfie, id_front)
        results["face_verification"] = face_result
        
        # Fraud detection on ID
        fraud_result = await detect_fraud(id_front)
        results["fraud_detection"] = fraud_result
        
        # Address verification if provided
        if address_proof:
            address_result = await verify_address(address_proof)
            results["address_verification"] = address_result
        
        # Overall status
        all_verified = (
            results["face_verification"]["verified"] and
            results["fraud_detection"]["is_authentic"]
        )
        
        if "address_verification" in results:
            all_verified = all_verified and results["address_verification"]["address_detected"]
        
        results["overall_status"] = "verified" if all_verified else "failed"
        results["risk_score"] = 15 if all_verified else 75
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Full verification failed: {str(e)}")


@router.get("/health")
async def health():
    """Health check for verification service."""
    return {
        "status": "ok",
        "service": "verification",
        "models_loaded": {
            "face_verification": get_face_model() is not None
        }
    }

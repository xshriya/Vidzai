from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Optional
import sys
from pathlib import Path

router = APIRouter(prefix="/ocr", tags=["OCR Service"])

# Lazy imports
_ocr_service = None
_text_classifier = None
TESSERACT_AVAILABLE = False


def get_ocr_service():
    global _ocr_service, TESSERACT_AVAILABLE
    if _ocr_service is None:
        try:
            face_detection_path = Path(__file__).parent.parent.parent.parent.parent / "Face detection" / "n1"
            sys.path.insert(0, str(face_detection_path))
            from ocr_service import extract_text, TESSERACT_AVAILABLE as tav
            _ocr_service = extract_text
            TESSERACT_AVAILABLE = tav
        except ImportError:
            _ocr_service = lambda x: "OCR service unavailable (Tesseract not installed)"
    return _ocr_service


def get_text_classifier():
    global _text_classifier
    if _text_classifier is None:
        try:
            face_detection_path = Path(__file__).parent.parent.parent.parent.parent / "Face detection" / "n1"
            sys.path.insert(0, str(face_detection_path))
            from text_classifier import classify_fields
            _text_classifier = classify_fields
        except ImportError:
            _text_classifier = lambda texts: [{"text": t, "field": "OTHER"} for t in texts]
    return _text_classifier


@router.post("/extract-text")
async def extract_text_from_image(file: UploadFile = File(...)):
    """
    Extract text from a document image using OCR.
    Returns extracted text content.
    """
    try:
        image_bytes = await file.read()
        extract_text = get_ocr_service()
        text = extract_text(image_bytes)
        
        return {
            "filename": file.filename,
            "text": text,
            "ocr_available": TESSERACT_AVAILABLE,
            "char_count": len(text)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


@router.post("/classify-fields")
async def classify_text_fields(texts: Optional[List[str]] = None, file: UploadFile = File(None)):
    """
    Classify extracted text into field types (NAME, DATE, ADDRESS, etc.)
    Accepts either a list of texts or an image file.
    """
    try:
        if file and file.filename:
            # Extract text from image first
            image_bytes = await file.read()
            extract_text = get_ocr_service()
            extracted = extract_text(image_bytes)
            texts = [line.strip() for line in extracted.split('\n') if line.strip()]
        
        if not texts:
            raise HTTPException(status_code=400, detail="No text provided")
        
        classify_fields = get_text_classifier()
        classified = classify_fields(texts)
        
        return {
            "fields": classified,
            "count": len(classified)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.get("/health")
async def health():
    """Health check for OCR service."""
    return {
        "status": "ok",
        "service": "ocr",
        "tesseract_available": TESSERACT_AVAILABLE
    }

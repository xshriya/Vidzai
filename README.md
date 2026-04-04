# Vidzai - AI-Powered Identity Verification & Fraud Detection for KYC Compliance

<div align="center">

**Identity Verification • Document Validation • Fraud Detection**

</div>

---

## Overview

Vidzai automates KYC verification using deep learning models for document classification, tamper detection, and identity verification.

---

## Features

- **Document Detection** - YOLOv8-based detection and classification of identity documents
- **Tamper Detection** - ResNet50 model for fraud detection
- **Address Verification** - Automated address field extraction from documents
- **OCR Integration** - Tesseract-powered text extraction
- **Web Dashboard** - Modern UI for KYC management

---

## Tech Stack

`Python` `PyTorch` `YOLOv8` `OpenCV` `FastAPI` `Tesseract OCR` `HTML/CSS/JS`

---

## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/Bhargavi872004/Vidzai_Digital_AI-Powered-Identity-Verification-and-Fraud-Detection-for-KYC-Compliance.git
cd Vidzai_Digital_AI-Powered-Identity-Verification-and-Fraud-Detection-for-KYC-Compliance
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Backend API
```bash
cd "document detection/document detection"
uvicorn main:app --reload --port 8000
```
Access API docs at: `http://localhost:8000/docs`

### 4. Open Frontend
Open `frontend/ai powered ekyc/index.html` in your browser.

### 5. Explore Notebooks (Optional)
```bash
jupyter notebook
```
Open `tampmodeltraining.ipynb` or `Adress_Detection.ipynb` to explore model training.

---

## Project Structure

```
Vidzai/
├── document detection/    # Backend API & ML models
├── Face detection/        # OCR & face verification
├── frontend/              # Web dashboard
├── *.ipynb                # Training notebooks
└── requirements.txt       # Dependencies
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/document/detect` | Detect documents in image |
| `POST /api/v1/verification/detect-fraud` | Tamper detection |
| `POST /api/v1/ocr/extract-text` | Text extraction via OCR |

Full API docs at `http://localhost:8000/docs`

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for secure digital identity verification**

</div>

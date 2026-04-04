from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import document, ocr, vereification

app = FastAPI(title="Vidzai KYC - AI-Powered Identity Verification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document.router,prefix="/api/v1")
app.include_router(ocr.router,prefix="/api/v1")
app.include_router(vereification.router,prefix="/api/v1")

@app.get("/health")
def health():
    return {"status":"ok"}

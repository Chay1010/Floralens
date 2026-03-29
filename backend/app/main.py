"""
FloraLens API — FastAPI application for plant/flower classification.

Endpoints:
    POST /predict         — Upload an image, get top-5 flower predictions
    GET  /health          — Health check & model readiness
    GET  /classes          — List all 102 flower class names
    GET  /                — Root info
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import (
    API_HOST,
    API_PORT,
    FLOWER_NAMES,
    MAX_IMAGE_SIZE_MB,
    NUM_CLASSES,
)
from app.model import FloraLensPredictor, get_predictor
from app.preprocessing import preprocess_image_bytes

logger = logging.getLogger("floralens")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")


# ──────────────────── Pydantic Schemas ────────────────────


class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predictions: List[Prediction]
    inference_time_ms: float
    model_status: str = "ready"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int


# ──────────────────── Lifespan ────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    logger.info("🌿 FloraLens starting up — loading model...")
    predictor = get_predictor()
    if predictor.is_ready:
        logger.info("✅ Model ready for inference.")
    else:
        logger.warning("⚠️  Model not loaded — /predict will return 503.")
    yield
    logger.info("🌿 FloraLens shutting down.")


# ──────────────────── App ────────────────────


app = FastAPI(
    title="FloraLens API",
    description="AI-powered plant & flower identification from photographs.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend to call the API from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────── Routes ────────────────────


@app.get("/", tags=["info"])
async def root():
    return {
        "app": "FloraLens",
        "version": "1.0.0",
        "docs": "/docs",
        "description": "Upload a flower/plant photo to /predict to identify the species.",
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
async def health_check():
    predictor = get_predictor()
    return HealthResponse(
        status="ok" if predictor.is_ready else "model_unavailable",
        model_loaded=predictor.is_ready,
        num_classes=NUM_CLASSES,
    )


@app.get("/classes", tags=["info"])
async def list_classes():
    return {"num_classes": NUM_CLASSES, "classes": FLOWER_NAMES}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(..., description="Image of a plant or flower")):
    """Classify an uploaded plant/flower image.

    Accepts JPEG/PNG images up to 10 MB.
    Returns top-5 predictions with confidence scores and inference latency.
    """
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Use JPEG, PNG, or WebP.",
        )

    # Read & validate size
    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({size_mb:.1f} MB). Max is {MAX_IMAGE_SIZE_MB} MB.",
        )

    # Check model readiness
    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please deploy the ONNX model first.",
        )

    # Preprocess
    try:
        input_array = preprocess_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

    # Inference with latency tracking
    predictions, latency_ms = predictor.predict_with_latency(input_array, top_k=5)

    logger.info(
        f"🌸 Prediction: {predictions[0]['class_name']} "
        f"({predictions[0]['confidence']:.1%}) — {latency_ms:.1f}ms"
    )

    return PredictionResponse(
        predictions=[Prediction(**p) for p in predictions],
        inference_time_ms=round(latency_ms, 2),
    )


# ──────────────────── Entry Point ────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True)

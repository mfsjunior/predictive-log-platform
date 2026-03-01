"""
Predictive Log Intelligence Platform — FastAPI Application.
Main entry point for the Python ML service.
"""
import os
import logging

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import train, predict, anomaly, monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Predictive Log Intelligence Platform — ML Service",
    description=(
        "Machine Learning service for web log analysis. "
        "Provides model training, error probability prediction, "
        "response time estimation, anomaly detection, and data drift monitoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics instrumentation — exposes /metrics endpoint
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health"],
    ).instrument(app).expose(app, include_in_schema=True, tags=["Monitoring"])
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed. Metrics endpoint disabled.")

# Include routers
app.include_router(train.router, tags=["Training"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(anomaly.router, tags=["Anomaly Detection"])
app.include_router(monitor.router, tags=["Monitoring"])

# WebSocket router for real-time alerts
from app.routers import websocket as ws_router
app.include_router(ws_router.router, tags=["WebSocket"])


@app.on_event("startup")
async def startup_event():
    """Load pre-trained models on startup if available."""
    logger.info("Starting Predictive Log Intelligence ML Service...")

    classifier_path = os.path.join(settings.MODELS_DIR, "best_classifier.joblib")
    regressor_path = os.path.join(settings.MODELS_DIR, "best_regressor.joblib")

    if os.path.exists(classifier_path):
        try:
            from app.models.classifier import ClassifierPipeline
            train.classifier_pipeline = ClassifierPipeline()
            train.classifier_pipeline.best_model = joblib.load(classifier_path)
            train.classifier_pipeline.best_model_name = "loaded_from_disk"
            logger.info(f"Loaded classifier from {classifier_path}")
        except Exception as e:
            logger.warning(f"Failed to load classifier: {e}")

    if os.path.exists(regressor_path):
        try:
            from app.models.regressor import RegressorPipeline
            train.regressor_pipeline = RegressorPipeline()
            train.regressor_pipeline.best_model = joblib.load(regressor_path)
            train.regressor_pipeline.best_model_name = "loaded_from_disk"
            train.regressor_pipeline.results["loaded_from_disk"] = {"rmse": 0.0}
            logger.info(f"Loaded regressor from {regressor_path}")
        except Exception as e:
            logger.warning(f"Failed to load regressor: {e}")

    logger.info("ML Service startup complete.")

    # Start periodic drift monitor
    try:
        from app.scheduler import start_scheduler
        start_scheduler()
    except Exception as e:
        logger.warning(f"Scheduler startup failed (non-critical): {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop scheduler on shutdown."""
    try:
        from app.scheduler import stop_scheduler
        stop_scheduler()
    except Exception:
        pass


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Predictive Log Intelligence Platform — ML Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "training": "POST /train",
            "generate_dataset": "POST /generate-dataset",
            "predict_error": "POST /predict/error",
            "predict_response_time": "POST /predict/response-time",
            "detect_anomaly": "POST /detect/anomaly",
            "drift_monitoring": "GET /monitor/drift",
            "model_health": "GET /monitor/health",
        },
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

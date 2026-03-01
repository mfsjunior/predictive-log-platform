"""
Prediction router — handles error probability and response time predictions.
"""
import time
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.feature_engineering import prepare_single_prediction, get_regression_features

logger = logging.getLogger(__name__)
router = APIRouter()


class ErrorPredictionRequest(BaseModel):
    method: str = Field(..., description="HTTP method (GET, POST, PUT, DELETE, PATCH)")
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    historical_avg_response: float = Field(
        ..., gt=0, description="Historical average response time in ms"
    )
    day_of_week: int = Field(
        default=2, ge=0, le=6,
        description="Day of the week (0=Monday, 6=Sunday)"
    )


class ErrorPredictionResponse(BaseModel):
    error_probability: float
    risk_level: str
    model_used: str | None = None
    inference_time_ms: float | None = None


class ResponseTimePredictionRequest(BaseModel):
    method: str = Field(..., description="HTTP method")
    hour: int = Field(..., ge=0, le=23, description="Hour of the day")
    historical_avg_response: float = Field(..., gt=0, description="Historical avg response time ms")
    day_of_week: int = Field(default=2, ge=0, le=6)
    is_error: int = Field(default=0, ge=0, le=1, description="Whether the request is an error")


class ResponseTimePredictionResponse(BaseModel):
    predicted_response_time_ms: float
    confidence_interval: dict
    model_used: str | None = None
    inference_time_ms: float | None = None


@router.post("/predict/error", response_model=ErrorPredictionResponse)
async def predict_error(request: ErrorPredictionRequest):
    """
    Predict the probability of an HTTP error (4xx/5xx).

    Returns error probability and risk level (LOW/MEDIUM/HIGH/CRITICAL).
    """
    from app.routers.train import classifier_pipeline

    if classifier_pipeline is None or classifier_pipeline.best_model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained classifier available. POST /train first."
        )

    start = time.time()

    try:
        X = prepare_single_prediction(
            method=request.method.upper(),
            hour=request.hour,
            historical_avg_response=request.historical_avg_response,
            day_of_week=request.day_of_week,
        )

        result = classifier_pipeline.predict_error_probability(X)
        inference_ms = (time.time() - start) * 1000

        return ErrorPredictionResponse(
            error_probability=result["error_probability"],
            risk_level=result["risk_level"],
            model_used=result.get("model_used"),
            inference_time_ms=round(inference_ms, 2),
        )
    except Exception as e:
        logger.error(f"Error prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/response-time", response_model=ResponseTimePredictionResponse)
async def predict_response_time(request: ResponseTimePredictionRequest):
    """
    Predict the expected response time in milliseconds.

    Returns predicted value with 95% confidence interval.
    """
    from app.routers.train import regressor_pipeline

    if regressor_pipeline is None or regressor_pipeline.best_model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained regressor available. POST /train first."
        )

    start = time.time()

    try:
        X_base = prepare_single_prediction(
            method=request.method.upper(),
            hour=request.hour,
            historical_avg_response=request.historical_avg_response,
            day_of_week=request.day_of_week,
        )
        # Add is_error feature for regression
        X_base["is_error"] = request.is_error

        # Ensure column order matches regression features
        reg_features = get_regression_features()
        X = X_base[reg_features]

        result = regressor_pipeline.predict_response_time(X)
        inference_ms = (time.time() - start) * 1000

        return ResponseTimePredictionResponse(
            predicted_response_time_ms=result["predicted_response_time_ms"],
            confidence_interval=result["confidence_interval"],
            model_used=result.get("model_used"),
            inference_time_ms=round(inference_ms, 2),
        )
    except Exception as e:
        logger.error(f"Response time prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""
Anomaly detection router — detects anomalous web log entries.
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.feature_engineering import prepare_single_prediction

logger = logging.getLogger(__name__)
router = APIRouter()


class AnomalyDetectionRequest(BaseModel):
    response_time_ms: float = Field(..., gt=0, description="Response time in milliseconds")
    method: str = Field(default="GET", description="HTTP method")
    hour: int = Field(default=12, ge=0, le=23, description="Hour of the day")
    day_of_week: int = Field(default=2, ge=0, le=6)


class AnomalyDetailScore(BaseModel):
    value: float | None = None
    score: float | None = None
    is_anomaly: bool
    threshold: float | None = None


class AnomalyDetails(BaseModel):
    z_score: AnomalyDetailScore
    isolation_forest: AnomalyDetailScore


class AnomalyDetectionResponse(BaseModel):
    is_anomaly: bool
    score: float
    details: AnomalyDetails


@router.post("/detect/anomaly", response_model=AnomalyDetectionResponse)
async def detect_anomaly(request: AnomalyDetectionRequest):
    """
    Detect whether a web log entry is anomalous.

    Uses both Z-score and Isolation Forest for detection.
    Returns combined anomaly score and individual method details.
    """
    from app.routers.train import anomaly_detector

    if anomaly_detector is None:
        raise HTTPException(
            status_code=503,
            detail="No trained anomaly detector available. POST /train first."
        )

    try:
        features_df = prepare_single_prediction(
            method=request.method.upper(),
            hour=request.hour,
            historical_avg_response=request.response_time_ms,
            day_of_week=request.day_of_week,
        )

        result = anomaly_detector.detect(
            response_time_ms=request.response_time_ms,
            hour=request.hour,
            method=request.method,
            features_df=features_df,
        )

        return AnomalyDetectionResponse(
            is_anomaly=result["is_anomaly"],
            score=result["score"],
            details=AnomalyDetails(
                z_score=AnomalyDetailScore(
                    value=result["details"]["z_score"]["value"],
                    is_anomaly=result["details"]["z_score"]["is_anomaly"],
                    threshold=result["details"]["z_score"]["threshold"],
                ),
                isolation_forest=AnomalyDetailScore(
                    score=result["details"]["isolation_forest"]["score"],
                    is_anomaly=result["details"]["isolation_forest"]["is_anomaly"],
                ),
            ),
        )
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

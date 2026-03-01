"""
Monitoring router — data drift detection and model health.
"""
import os
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import create_engine

from app.config import settings
from app.feature_engineering import engineer_features
from app.monitoring.drift import generate_drift_report, get_reference_data

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/monitor/drift")
async def check_drift():
    """
    Check for data drift between training data and current production data.

    Compares the current database contents against the training reference
    using Evidently AI statistical tests. Generates an HTML report.
    """
    reference = get_reference_data()
    if reference is None:
        raise HTTPException(
            status_code=503,
            detail="No reference data available. Train a model first (POST /train)."
        )

    try:
        # Fetch current data from database
        current_df = _fetch_current_data()

        if current_df is None or len(current_df) == 0:
            raise HTTPException(
                status_code=404,
                detail="No current data available in database."
            )

        # Apply feature engineering to current data
        current_features = engineer_features(current_df)

        # Generate drift report
        result = generate_drift_report(
            current_data=current_features,
            output_dir=os.path.join(settings.DATA_DIR, "drift_reports"),
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drift monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor/retrain-if-drift")
async def retrain_if_drift():
    """
    Check drift and retrain models if drift exceeds 50% threshold.
    Returns drift status and whether retraining was triggered.
    """
    reference = get_reference_data()
    if reference is None:
        raise HTTPException(status_code=503, detail="No reference data. Train first (POST /train).")

    try:
        current_df = _fetch_current_data()
        if current_df is None or len(current_df) == 0:
            return {"drift_detected": False, "retrained": False, "reason": "No current data available."}

        from app.feature_engineering import engineer_features
        current_features = engineer_features(current_df)

        result = generate_drift_report(current_data=current_features)
        dataset_drift = result.get("dataset_drift", False)
        drift_share = result.get("drift_share", 0.0)

        if dataset_drift:
            logger.warning(f"Drift detected (share={drift_share:.2%}). Triggering retraining...")
            from app.routers.train import train_models
            train_result = await train_models()
            return {
                "drift_detected": True,
                "drift_share": drift_share,
                "retrained": True,
                "training_result": {
                    "status": train_result.status,
                    "training_time_seconds": train_result.training_time_seconds,
                    "best_classifier": train_result.classifier_results.get("best_model"),
                    "best_regressor": train_result.regressor_results.get("best_model"),
                },
            }
        else:
            return {
                "drift_detected": False,
                "drift_share": drift_share,
                "retrained": False,
                "reason": "No significant drift detected. Models are up to date.",
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrain-if-drift failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/drift/report")
async def get_drift_report():
    """
    Download the latest drift report as HTML.
    """
    drift_dir = os.path.join(settings.DATA_DIR, "drift_reports")
    if not os.path.exists(drift_dir):
        raise HTTPException(status_code=404, detail="No drift reports available.")

    # Find latest report
    reports = sorted(
        [f for f in os.listdir(drift_dir) if f.endswith(".html")],
        reverse=True,
    )
    if not reports:
        raise HTTPException(status_code=404, detail="No drift reports generated yet.")

    report_path = os.path.join(drift_dir, reports[0])
    return FileResponse(
        report_path,
        media_type="text/html",
        filename=reports[0],
    )


@router.get("/monitor/health")
async def model_health():
    """
    Return current model health status and metadata.
    """
    from app.routers.train import classifier_pipeline, regressor_pipeline, anomaly_detector

    classifier_status = "loaded" if (classifier_pipeline and classifier_pipeline.best_model) else "not_loaded"
    regressor_status = "loaded" if (regressor_pipeline and regressor_pipeline.best_model) else "not_loaded"
    anomaly_status = "loaded" if anomaly_detector else "not_loaded"

    models_dir = settings.MODELS_DIR
    classifier_file = os.path.join(models_dir, "best_classifier.joblib")
    regressor_file = os.path.join(models_dir, "best_regressor.joblib")

    return {
        "status": "healthy" if classifier_status == "loaded" else "degraded",
        "models": {
            "classifier": {
                "status": classifier_status,
                "name": getattr(classifier_pipeline, "best_model_name", None) if classifier_pipeline else None,
                "file_exists": os.path.exists(classifier_file),
            },
            "regressor": {
                "status": regressor_status,
                "name": getattr(regressor_pipeline, "best_model_name", None) if regressor_pipeline else None,
                "file_exists": os.path.exists(regressor_file),
            },
            "anomaly_detector": {
                "status": anomaly_status,
            },
        },
        "reference_data_available": get_reference_data() is not None,
    }


def _fetch_current_data() -> pd.DataFrame | None:
    """Fetch current data from database."""
    try:
        engine = create_engine(settings.DATABASE_URL)
        query = "SELECT * FROM web_logs ORDER BY timestamp DESC LIMIT 5000"
        df = pd.read_sql(query, engine)
        return df if len(df) > 0 else None
    except Exception as e:
        logger.warning(f"Failed to fetch current data: {e}")
        return None

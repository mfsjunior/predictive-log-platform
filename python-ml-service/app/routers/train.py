"""
Training router — handles model training pipeline.
POST /train triggers full ML pipeline: data fetch, feature engineering,
model training, evaluation, MLflow logging (via MlflowTracker), and model persistence.
"""
import os
import time
import logging
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine
from pydantic import BaseModel

from app.config import settings
from app.dataset_generator import generate_synthetic_dataset, save_dataset
from app.feature_engineering import (
    engineer_features,
    get_classification_features,
    get_regression_features,
)
from app.models.classifier import ClassifierPipeline
from app.models.regressor import RegressorPipeline
from app.models.anomaly import AnomalyDetector
from app.monitoring.drift import set_reference_data
from app.visualization.plots import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_shap_values,
)
from app.infrastructure.model_registry import ModelRegistry
from app.infrastructure.mlflow_tracker import MlflowTracker

logger = logging.getLogger(__name__)
router = APIRouter()

# Infrastructure: model registry (replaces global mutable vars)
registry = ModelRegistry.instance()

# Infrastructure: MLflow tracker (with retry + health check)
tracker = MlflowTracker(
    tracking_uri=settings.MLFLOW_TRACKING_URI,
    experiment_name=settings.EXPERIMENT_NAME,
)


class TrainResponse(BaseModel):
    status: str
    training_time_seconds: float
    num_samples: int
    classifier_results: dict
    regressor_results: dict
    anomaly_detector_fitted: bool
    plots_generated: list[str]
    model_version: str


class GenerateDatasetResponse(BaseModel):
    status: str
    num_records: int
    file_path: str
    error_rate: float
    avg_response_time: float


@router.post("/generate-dataset", response_model=GenerateDatasetResponse)
async def generate_dataset():
    """Generate synthetic dataset and save to CSV + database."""
    try:
        df = generate_synthetic_dataset(n_records=5000)
        filepath = os.path.join(settings.DATA_DIR, "web_logs.csv")
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        save_dataset(df, filepath)

        # Also insert into database
        try:
            engine = create_engine(settings.DATABASE_URL)
            db_df = df.copy()
            db_df.to_sql("web_logs", engine, if_exists="append", index=False)
            logger.info(f"Inserted {len(df)} records into database")
        except Exception as e:
            logger.warning(f"Database insert failed (will continue): {e}")

        return GenerateDatasetResponse(
            status="success",
            num_records=len(df),
            file_path=filepath,
            error_rate=float((df["status_code"] >= 400).mean()),
            avg_response_time=float(df["response_time_ms"].mean()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainResponse)
async def train_models():
    """
    Full training pipeline:
    1. Fetch data from database (fallback to CSV)
    2. Feature engineering
    3. Train classifiers and regressors
    4. Evaluate all models
    5. Log to MLflow
    6. Persist best models
    7. Generate evaluation plots
    """
    global classifier_pipeline, regressor_pipeline, anomaly_detector

    start_time = time.time()
    model_version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    plots_generated = []

    try:
        # 1. Fetch data
        df = _fetch_training_data()
        logger.info(f"Training with {len(df)} records")

        # 2. Feature engineering
        df_features = engineer_features(df)

        # Store reference data for drift monitoring
        set_reference_data(df_features)

        # 3. Prepare features and targets
        clf_features = get_classification_features()
        reg_features = get_regression_features()

        X_clf = df_features[clf_features]
        y_clf = df_features["is_error"]

        X_reg = df_features[reg_features]
        y_reg = df_features["response_time_ms"]

        # 4. Train/test split
        from sklearn.model_selection import train_test_split

        X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
            X_clf, y_clf, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE,
            stratify=y_clf,
        )

        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
            X_reg, y_reg, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE,
        )

        # 5. Train classifiers (domain layer — pure, no framework)
        classifier_pipeline = ClassifierPipeline(random_state=settings.RANDOM_STATE)
        clf_results = classifier_pipeline.train_and_evaluate(
            X_clf_train, X_clf_test, y_clf_train, y_clf_test
        )

        # 6. Train regressors (domain layer)
        regressor_pipeline = RegressorPipeline(random_state=settings.RANDOM_STATE)
        reg_results = regressor_pipeline.train_and_evaluate(
            X_reg_train, X_reg_test, y_reg_train, y_reg_test
        )

        # 7. Fit anomaly detector
        anomaly_detector = AnomalyDetector()
        anomaly_detector.fit(df_features, clf_features)

        # 8. Register models in registry (replaces globals)
        registry.register("classifier", classifier_pipeline, {"version": model_version})
        registry.register("regressor", regressor_pipeline, {"version": model_version})
        registry.register("anomaly_detector", anomaly_detector, {"version": model_version})

        # 9. Save best models to disk
        os.makedirs(settings.MODELS_DIR, exist_ok=True)
        classifier_pipeline.save_best_model(
            os.path.join(settings.MODELS_DIR, "best_classifier.joblib")
        )
        regressor_pipeline.save_best_model(
            os.path.join(settings.MODELS_DIR, "best_regressor.joblib")
        )

        # 10. Log to MLflow via infrastructure tracker (with retry + health check)
        for name, model in classifier_pipeline.models.items():
            tracker.log_classifier_run(
                name=name, model=model,
                metrics=classifier_pipeline.results[name],
                is_best=(name == classifier_pipeline.best_model_name),
            )
        for name, model in regressor_pipeline.models.items():
            tracker.log_regressor_run(
                name=name, model=model,
                metrics=regressor_pipeline.results[name],
                is_best=(name == regressor_pipeline.best_model_name),
            )

        # 10. Generate plots
        plots_generated = _generate_plots(
            classifier_pipeline, X_clf_test, clf_features
        )

        training_time = time.time() - start_time

        return TrainResponse(
            status="success",
            training_time_seconds=round(training_time, 2),
            num_samples=len(df),
            classifier_results=clf_results,
            regressor_results=reg_results,
            anomaly_detector_fitted=True,
            plots_generated=plots_generated,
            model_version=model_version,
        )

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


def _fetch_training_data() -> pd.DataFrame:
    """Fetch training data from database, fallback to CSV."""
    try:
        engine = create_engine(settings.DATABASE_URL)
        query = "SELECT * FROM web_logs ORDER BY timestamp"
        df = pd.read_sql(query, engine)
        if len(df) > 0:
            logger.info(f"Fetched {len(df)} records from database")
            return df
    except Exception as e:
        logger.warning(f"Database fetch failed: {e}")

    # Fallback to CSV
    csv_path = os.path.join(settings.DATA_DIR, "web_logs.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from CSV: {csv_path}")
        return df

    # Generate new dataset
    logger.info("No data found. Generating synthetic dataset.")
    df = generate_synthetic_dataset(n_records=5000)
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    save_dataset(df, csv_path)
    return df


def _generate_plots(
    clf_pipeline: ClassifierPipeline,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> list[str]:
    """Generate all evaluation plots."""
    plots = []
    plots_dir = os.path.join(settings.DATA_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        # ROC Curve
        fpr_dict = {name: r["fpr"] for name, r in clf_pipeline.results.items()}
        tpr_dict = {name: r["tpr"] for name, r in clf_pipeline.results.items()}
        auc_dict = {name: r["roc_auc"] for name, r in clf_pipeline.results.items()}
        roc_path = plot_roc_curve(fpr_dict, tpr_dict, auc_dict,
                                  os.path.join(plots_dir, "roc_curve.png"))
        plots.append(roc_path)
    except Exception as e:
        logger.warning(f"ROC curve plot failed: {e}")

    try:
        # Confusion Matrix for best model
        best_name = clf_pipeline.best_model_name
        cm = clf_pipeline.results[best_name]["confusion_matrix"]
        cm_path = plot_confusion_matrix(cm, best_name,
                                         os.path.join(plots_dir, "confusion_matrix.png"))
        plots.append(cm_path)
    except Exception as e:
        logger.warning(f"Confusion matrix plot failed: {e}")

    try:
        # Feature importance
        best_model = clf_pipeline.best_model
        if hasattr(best_model, "feature_importances_"):
            fi_path = plot_feature_importance(
                feature_names, best_model.feature_importances_, best_name,
                os.path.join(plots_dir, "feature_importance.png"),
            )
            plots.append(fi_path)
    except Exception as e:
        logger.warning(f"Feature importance plot failed: {e}")

    try:
        # SHAP values
        shap_path = plot_shap_values(
            clf_pipeline.best_model, X_test,
            os.path.join(plots_dir, "shap_values.png"),
        )
        plots.append(shap_path)
    except Exception as e:
        logger.warning(f"SHAP plot failed: {e}")

    return plots

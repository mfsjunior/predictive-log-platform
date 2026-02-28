"""
Regression models for response time prediction.
Trains and compares Linear Regression, Random Forest Regressor, and Gradient Boosting.
Evaluates using RMSE, MAE, and R².
"""
import os
import logging
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.config import settings

logger = logging.getLogger(__name__)


class RegressorPipeline:
    """Pipeline for training and evaluating regression models."""

    def __init__(self):
        self.models: dict[str, Any] = {}
        self.results: dict[str, dict] = {}
        self.best_model_name: str | None = None
        self.best_model: Any = None

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> dict:
        """
        Train all regressors, evaluate, and select the best one.

        Returns:
            Dictionary with model comparison results and best model info.
        """
        regressors = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=settings.RANDOM_STATE,
            ),
        }

        for name, model in regressors.items():
            logger.info(f"Training regressor: {name}")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            self.models[name] = model
            self.results[name] = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "y_test": y_test.values.tolist(),
                "y_pred": y_pred.tolist(),
            }

            logger.info(f"  {name} — RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

        # Select best model by R²
        self.best_model_name = max(self.results, key=lambda k: self.results[k]["r2"])
        self.best_model = self.models[self.best_model_name]

        logger.info(f"Best regressor: {self.best_model_name} "
                     f"(R²={self.results[self.best_model_name]['r2']:.4f})")

        return {
            "best_model": self.best_model_name,
            "models": {k: {
                "rmse": v["rmse"],
                "mae": v["mae"],
                "r2": v["r2"],
            } for k, v in self.results.items()},
        }

    def save_best_model(self, filepath: str) -> str:
        """Save the best model to disk using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.best_model, filepath)
        logger.info(f"Saved best regressor to {filepath}")
        return filepath

    def log_to_mlflow(self, experiment_name: str) -> str | None:
        """Log all models and metrics to MLflow."""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"MLflow tracking not available: {e}")
            return None

        run_id = None
        for name, model in self.models.items():
            try:
                with mlflow.start_run(run_name=f"regressor_{name}") as run:
                    if name == self.best_model_name:
                        run_id = run.info.run_id
                    metrics = self.results[name]
                    mlflow.log_param("model_type", "regressor")
                    mlflow.log_param("model_name", name)
                    mlflow.log_metric("rmse", metrics["rmse"])
                    mlflow.log_metric("mae", metrics["mae"])
                    mlflow.log_metric("r2", metrics["r2"])
                    mlflow.log_param("is_best", name == self.best_model_name)

                    mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception as e:
                logger.warning(f"Failed to log {name} to MLflow: {e}")

        return run_id

    def predict_response_time(self, X: pd.DataFrame) -> dict:
        """
        Predict response time with confidence interval.

        Args:
            X: Feature DataFrame for prediction.

        Returns:
            Dictionary with predicted_ms, confidence_interval_lower,
            confidence_interval_upper, and model info.
        """
        if self.best_model is None:
            raise RuntimeError("No trained model available. Train first.")

        prediction = float(self.best_model.predict(X)[0])

        # Estimate confidence interval using residual standard deviation
        # This is a simplified approach; for production, use quantile regression
        best_metrics = self.results.get(self.best_model_name, {})
        rmse = best_metrics.get("rmse", prediction * 0.2)

        # 95% confidence interval (±1.96 * RMSE as proxy)
        ci_margin = 1.96 * rmse
        lower = max(0.0, prediction - ci_margin)
        upper = prediction + ci_margin

        return {
            "predicted_response_time_ms": round(prediction, 2),
            "confidence_interval": {
                "lower_bound_ms": round(lower, 2),
                "upper_bound_ms": round(upper, 2),
                "confidence_level": 0.95,
            },
            "model_used": self.best_model_name,
        }


def load_regressor(filepath: str) -> RegressorPipeline:
    """Load a saved regressor pipeline."""
    pipeline = RegressorPipeline()
    pipeline.best_model = joblib.load(filepath)
    pipeline.best_model_name = "loaded_model"
    return pipeline

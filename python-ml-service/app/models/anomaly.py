"""
Anomaly detection module using Z-score and Isolation Forest.
"""
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from app.config import settings

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Anomaly detection using Z-score and Isolation Forest."""

    def __init__(self):
        self.isolation_forest: IsolationForest | None = None
        self.training_stats: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, features: list[str]) -> dict:
        """
        Fit the anomaly detection models on training data.

        Args:
            df: Training DataFrame with feature columns.
            features: List of feature column names to use.

        Returns:
            Dictionary with fitting statistics.
        """
        X = df[features].values

        # Compute Z-score reference statistics
        self.training_stats = {
            "mean": np.mean(X, axis=0).tolist(),
            "std": np.std(X, axis=0).tolist(),
            "n_features": len(features),
            "n_samples": len(df),
        }

        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=settings.RANDOM_STATE,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X)

        logger.info(f"Anomaly detector fitted on {len(df)} samples with {len(features)} features")

        return {
            "n_samples": len(df),
            "n_features": len(features),
            "contamination": 0.05,
        }

    def detect(
        self,
        response_time_ms: float,
        hour: int,
        method: str,
        features_df: pd.DataFrame | None = None,
    ) -> dict:
        """
        Detect anomaly for a single data point.

        Uses both Z-score and Isolation Forest, combining their results.

        Args:
            response_time_ms: Response time in milliseconds.
            hour: Hour of the day.
            method: HTTP method.
            features_df: Optional pre-built feature DataFrame.

        Returns:
            Dictionary with is_anomaly flag, anomaly score, and details.
        """
        # Z-score anomaly detection on response_time
        z_score_result = self._zscore_detect(response_time_ms)

        # Isolation Forest detection
        if_result = self._isolation_forest_detect(features_df)

        # Combined decision: anomaly if either method flags it
        is_anomaly = z_score_result["is_anomaly"] or if_result["is_anomaly"]
        combined_score = (z_score_result["z_score_normalized"] + if_result["score"]) / 2

        return {
            "is_anomaly": bool(is_anomaly),
            "score": round(float(combined_score), 4),
            "details": {
                "z_score": {
                    "value": round(float(z_score_result["z_score"]), 4),
                    "is_anomaly": bool(z_score_result["is_anomaly"]),
                    "threshold": 3.0,
                },
                "isolation_forest": {
                    "score": round(float(if_result["raw_score"]), 4),
                    "is_anomaly": bool(if_result["is_anomaly"]),
                },
            },
        }

    def _zscore_detect(self, response_time_ms: float, threshold: float = 3.0) -> dict:
        """Detect anomaly using Z-score on response time."""
        if not self.training_stats:
            # Fallback with reasonable defaults
            mean_rt = 250.0
            std_rt = 150.0
        else:
            # Use the first feature (response time proxy from rolling avg)
            mean_rt = self.training_stats["mean"][0] if self.training_stats["mean"] else 250.0
            std_rt = self.training_stats["std"][0] if self.training_stats["std"] else 150.0

        std_rt = max(std_rt, 1e-6)  # Avoid division by zero
        z_score = (response_time_ms - mean_rt) / std_rt
        is_anomaly = abs(z_score) > threshold

        # Normalize z-score to [-1, 1] range for combination
        z_normalized = -1.0 if z_score > threshold else (1.0 if z_score < -threshold else 0.0)

        return {
            "z_score": z_score,
            "z_score_normalized": z_normalized,
            "is_anomaly": is_anomaly,
        }

    def _isolation_forest_detect(self, features_df: pd.DataFrame | None) -> dict:
        """Detect anomaly using Isolation Forest."""
        if self.isolation_forest is None or features_df is None:
            return {
                "raw_score": 0.0,
                "score": 0.0,
                "is_anomaly": False,
            }

        X = features_df.values
        score = self.isolation_forest.decision_function(X)[0]
        prediction = self.isolation_forest.predict(X)[0]

        return {
            "raw_score": float(score),
            "score": float(score),
            "is_anomaly": prediction == -1,
        }

    def detect_batch(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """
        Detect anomalies in a batch of records.

        Returns:
            DataFrame with added columns: is_anomaly, anomaly_score.
        """
        result_df = df.copy()

        if self.isolation_forest is None:
            result_df["is_anomaly"] = False
            result_df["anomaly_score"] = 0.0
            return result_df

        X = df[features].values

        # Isolation Forest scores
        if_scores = self.isolation_forest.decision_function(X)
        if_predictions = self.isolation_forest.predict(X)

        # Z-score on response_time_ms
        if "response_time_ms" in df.columns:
            z_scores = np.abs(stats.zscore(df["response_time_ms"]))
            z_anomaly = z_scores > 3.0
        else:
            z_anomaly = np.zeros(len(df), dtype=bool)

        result_df["is_anomaly"] = (if_predictions == -1) | z_anomaly
        result_df["anomaly_score"] = if_scores

        return result_df

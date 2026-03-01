"""
Classification models for error probability prediction.
Trains and compares Logistic Regression, Random Forest, and XGBoost.
Selects best model based on ROC-AUC, F1-score, and Precision-Recall.

DOMAIN LAYER — no framework dependencies (no mlflow, no fastapi, no settings).
"""
import os
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class ClassifierPipeline:
    """Pipeline for training and evaluating classification models."""

    def __init__(self, random_state: int = 42):
        self.models: dict[str, Any] = {}
        self.results: dict[str, dict] = {}
        self.best_model_name: str | None = None
        self.best_model: Any = None
        self._random_state = random_state

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> dict:
        """
        Train all classifiers, evaluate, and select the best one.

        Returns:
            Dictionary with model comparison results and best model info.
        """
        classifiers = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=self._random_state,
                class_weight="balanced",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=self._random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "xgboost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self._random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=5.0,
            ),
        }

        for name, model in classifiers.items():
            logger.info(f"Training classifier: {name}")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred, average="weighted")
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)

            self.models[name] = model
            self.results[name] = {
                "roc_auc": float(roc_auc),
                "f1_score": float(f1),
                "accuracy": float(accuracy),
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "precision_curve": precision_vals.tolist(),
                "recall_curve": recall_vals.tolist(),
            }

            logger.info(f"  {name} — ROC-AUC={roc_auc:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")

        # Select best model by ROC-AUC
        self.best_model_name = max(self.results, key=lambda k: self.results[k]["roc_auc"])
        self.best_model = self.models[self.best_model_name]

        logger.info(f"Best classifier: {self.best_model_name} "
                     f"(ROC-AUC={self.results[self.best_model_name]['roc_auc']:.4f})")

        return {
            "best_model": self.best_model_name,
            "models": {k: {
                "roc_auc": v["roc_auc"],
                "f1_score": v["f1_score"],
                "accuracy": v["accuracy"],
            } for k, v in self.results.items()},
        }

    def save_best_model(self, filepath: str) -> str:
        """Save the best model to disk using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.best_model, filepath)
        logger.info(f"Saved best classifier to {filepath}")
        return filepath

    # MLflow logging removed from domain — use MlflowTracker in infrastructure layer

    def predict_error_probability(self, X: pd.DataFrame) -> dict:
        """
        Predict the probability of an error (4xx/5xx).

        Args:
            X: Feature DataFrame for a single or batch prediction.

        Returns:
            Dictionary with error_probability and risk_level.
        """
        if self.best_model is None:
            raise RuntimeError("No trained model available. Train first.")

        proba = self.best_model.predict_proba(X)[:, 1]
        prob_val = float(proba[0])

        if prob_val < 0.15:
            risk = "LOW"
        elif prob_val < 0.40:
            risk = "MEDIUM"
        elif prob_val < 0.70:
            risk = "HIGH"
        else:
            risk = "CRITICAL"

        return {
            "error_probability": round(prob_val, 4),
            "risk_level": risk,
            "model_used": self.best_model_name,
        }


def load_classifier(filepath: str) -> ClassifierPipeline:
    """Load a saved classifier pipeline."""
    pipeline = ClassifierPipeline()
    pipeline.best_model = joblib.load(filepath)
    pipeline.best_model_name = "loaded_model"
    return pipeline

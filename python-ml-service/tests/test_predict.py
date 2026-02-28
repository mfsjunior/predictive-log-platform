"""
Tests for prediction endpoints using FastAPI TestClient.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dataset_generator import generate_synthetic_dataset
from app.feature_engineering import (
    engineer_features,
    get_classification_features,
    get_regression_features,
)
from app.models.classifier import ClassifierPipeline
from app.models.regressor import RegressorPipeline
from app.models.anomaly import AnomalyDetector
from app.routers import train


client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_models():
    """Set up trained models for prediction tests."""
    df = generate_synthetic_dataset(n_records=500, seed=42)
    df_feat = engineer_features(df)

    clf_features = get_classification_features()
    reg_features = get_regression_features()

    from sklearn.model_selection import train_test_split

    X_clf = df_feat[clf_features]
    y_clf = df_feat["is_error"]
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    X_reg = df_feat[reg_features]
    y_reg = df_feat["response_time_ms"]
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Train classifier
    train.classifier_pipeline = ClassifierPipeline()
    train.classifier_pipeline.train_and_evaluate(
        X_clf_train, X_clf_test, y_clf_train, y_clf_test
    )

    # Train regressor
    train.regressor_pipeline = RegressorPipeline()
    train.regressor_pipeline.train_and_evaluate(
        X_reg_train, X_reg_test, y_reg_train, y_reg_test
    )

    # Train anomaly detector
    train.anomaly_detector = AnomalyDetector()
    train.anomaly_detector.fit(df_feat, clf_features)

    yield

    # Cleanup
    train.classifier_pipeline = None
    train.regressor_pipeline = None
    train.anomaly_detector = None


class TestRootEndpoint:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestErrorPrediction:
    def test_predict_error_valid(self):
        response = client.post("/predict/error", json={
            "method": "GET",
            "hour": 14,
            "historical_avg_response": 240.0,
        })
        assert response.status_code == 200
        data = response.json()
        assert "error_probability" in data
        assert 0 <= data["error_probability"] <= 1
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_predict_error_post_method(self):
        response = client.post("/predict/error", json={
            "method": "POST",
            "hour": 10,
            "historical_avg_response": 500.0,
        })
        assert response.status_code == 200
        data = response.json()
        assert "error_probability" in data

    def test_predict_error_invalid_hour(self):
        response = client.post("/predict/error", json={
            "method": "GET",
            "hour": 25,
            "historical_avg_response": 240.0,
        })
        assert response.status_code == 422

    def test_predict_error_negative_response(self):
        response = client.post("/predict/error", json={
            "method": "GET",
            "hour": 12,
            "historical_avg_response": -10.0,
        })
        assert response.status_code == 422


class TestResponseTimePrediction:
    def test_predict_response_time_valid(self):
        response = client.post("/predict/response-time", json={
            "method": "GET",
            "hour": 14,
            "historical_avg_response": 240.0,
        })
        assert response.status_code == 200
        data = response.json()
        assert "predicted_response_time_ms" in data
        assert data["predicted_response_time_ms"] > 0
        assert "confidence_interval" in data
        ci = data["confidence_interval"]
        assert ci["lower_bound_ms"] >= 0
        assert ci["upper_bound_ms"] > ci["lower_bound_ms"]
        assert ci["confidence_level"] == 0.95

    def test_predict_response_time_high_load(self):
        response = client.post("/predict/response-time", json={
            "method": "POST",
            "hour": 17,
            "historical_avg_response": 800.0,
            "is_error": 0,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_response_time_ms"] > 0


class TestModelHealth:
    def test_model_health(self):
        response = client.get("/monitor/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models" in data

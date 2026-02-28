"""
Tests for anomaly detection.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dataset_generator import generate_synthetic_dataset
from app.feature_engineering import (
    engineer_features,
    get_classification_features,
    prepare_single_prediction,
)
from app.models.anomaly import AnomalyDetector
from app.routers import train


client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_anomaly_detector():
    """Set up trained anomaly detector."""
    df = generate_synthetic_dataset(n_records=500, seed=42)
    df_feat = engineer_features(df)
    features = get_classification_features()

    # Also need classifier for the endpoint to work
    from app.models.classifier import ClassifierPipeline
    from sklearn.model_selection import train_test_split

    X = df_feat[features]
    y = df_feat["is_error"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train.classifier_pipeline = ClassifierPipeline()
    train.classifier_pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)

    train.anomaly_detector = AnomalyDetector()
    train.anomaly_detector.fit(df_feat, features)

    yield

    train.anomaly_detector = None
    train.classifier_pipeline = None


class TestAnomalyDetectorUnit:
    """Unit tests for AnomalyDetector class."""

    def test_fit(self):
        df = generate_synthetic_dataset(n_records=200, seed=42)
        df_feat = engineer_features(df)
        features = get_classification_features()

        detector = AnomalyDetector()
        result = detector.fit(df_feat, features)
        assert result["n_samples"] == 200
        assert result["n_features"] == len(features)

    def test_detect_normal(self):
        df = generate_synthetic_dataset(n_records=300, seed=42)
        df_feat = engineer_features(df)
        features = get_classification_features()

        detector = AnomalyDetector()
        detector.fit(df_feat, features)

        features_input = prepare_single_prediction("GET", 12, 200.0)
        result = detector.detect(
            response_time_ms=200.0, hour=12, method="GET",
            features_df=features_input,
        )
        assert "is_anomaly" in result
        assert "score" in result
        assert "details" in result

    def test_detect_extreme_anomaly(self):
        df = generate_synthetic_dataset(n_records=300, seed=42)
        df_feat = engineer_features(df)
        features = get_classification_features()

        detector = AnomalyDetector()
        detector.fit(df_feat, features)

        features_input = prepare_single_prediction("GET", 3, 15000.0)
        result = detector.detect(
            response_time_ms=15000.0, hour=3, method="GET",
            features_df=features_input,
        )
        # Extremely high response time should be flagged
        assert result["is_anomaly"] is True

    def test_batch_detection(self):
        df = generate_synthetic_dataset(n_records=300, seed=42)
        df_feat = engineer_features(df)
        features = get_classification_features()

        detector = AnomalyDetector()
        detector.fit(df_feat, features)

        result_df = detector.detect_batch(df_feat, features)
        assert "is_anomaly" in result_df.columns
        assert "anomaly_score" in result_df.columns
        assert result_df["is_anomaly"].sum() > 0  # Should find some anomalies


class TestAnomalyEndpoint:
    """Tests for the /detect/anomaly endpoint."""

    def test_detect_anomaly_normal(self):
        response = client.post("/detect/anomaly", json={
            "response_time_ms": 200.0,
            "method": "GET",
            "hour": 12,
        })
        assert response.status_code == 200
        data = response.json()
        assert "is_anomaly" in data
        assert "score" in data
        assert "details" in data

    def test_detect_anomaly_extreme(self):
        response = client.post("/detect/anomaly", json={
            "response_time_ms": 15000.0,
            "method": "GET",
            "hour": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["is_anomaly"] is True

    def test_detect_anomaly_invalid_input(self):
        response = client.post("/detect/anomaly", json={
            "response_time_ms": -5.0,
            "method": "GET",
            "hour": 12,
        })
        assert response.status_code == 422

    def test_detect_anomaly_all_methods(self):
        for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            response = client.post("/detect/anomaly", json={
                "response_time_ms": 300.0,
                "method": method,
                "hour": 10,
            })
            assert response.status_code == 200

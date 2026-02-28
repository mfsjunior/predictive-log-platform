"""
Tests for the ML training pipeline.
Validates dataset generation, feature engineering, and model training.
"""
import numpy as np
import pandas as pd
import pytest

from app.dataset_generator import generate_synthetic_dataset
from app.feature_engineering import (
    engineer_features,
    get_classification_features,
    get_regression_features,
    prepare_single_prediction,
)
from app.models.classifier import ClassifierPipeline
from app.models.regressor import RegressorPipeline
from app.models.anomaly import AnomalyDetector


class TestDatasetGenerator:
    """Tests for synthetic dataset generation."""

    def test_generates_correct_number_of_records(self):
        df = generate_synthetic_dataset(n_records=100, seed=42)
        assert len(df) == 100

    def test_has_required_columns(self):
        df = generate_synthetic_dataset(n_records=50, seed=42)
        required_cols = [
            "timestamp", "method", "path", "status_code",
            "response_time_ms", "user_agent", "ip_address", "bytes_sent",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_status_codes_are_valid(self):
        df = generate_synthetic_dataset(n_records=500, seed=42)
        valid_codes = {200, 201, 204, 301, 302, 400, 401, 403, 404, 500, 502, 503}
        assert set(df["status_code"].unique()).issubset(valid_codes)

    def test_methods_are_valid(self):
        df = generate_synthetic_dataset(n_records=500, seed=42)
        valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH"}
        assert set(df["method"].unique()).issubset(valid_methods)

    def test_response_times_positive(self):
        df = generate_synthetic_dataset(n_records=500, seed=42)
        assert (df["response_time_ms"] > 0).all()

    def test_contains_errors(self):
        df = generate_synthetic_dataset(n_records=1000, seed=42)
        error_rate = (df["status_code"] >= 400).mean()
        assert 0.05 < error_rate < 0.40, f"Error rate {error_rate:.2%} out of range"

    def test_contains_outliers(self):
        df = generate_synthetic_dataset(n_records=5000, seed=42)
        extreme_responses = df["response_time_ms"] > 3000
        assert extreme_responses.sum() > 0, "No outliers found"

    def test_reproducibility(self):
        df1 = generate_synthetic_dataset(n_records=100, seed=42)
        df2 = generate_synthetic_dataset(n_records=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestFeatureEngineering:
    """Tests for feature engineering pipeline."""

    @pytest.fixture
    def sample_df(self):
        return generate_synthetic_dataset(n_records=200, seed=42)

    def test_engineer_features_adds_columns(self, sample_df):
        result = engineer_features(sample_df)
        expected_cols = [
            "hour", "day_of_week", "is_business_hours",
            "rolling_avg_response", "hourly_frequency", "is_error",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing feature: {col}"

    def test_hour_range(self, sample_df):
        result = engineer_features(sample_df)
        assert result["hour"].between(0, 23).all()

    def test_day_of_week_range(self, sample_df):
        result = engineer_features(sample_df)
        assert result["day_of_week"].between(0, 6).all()

    def test_is_error_binary(self, sample_df):
        result = engineer_features(sample_df)
        assert set(result["is_error"].unique()).issubset({0, 1})

    def test_is_error_matches_status_code(self, sample_df):
        result = engineer_features(sample_df)
        expected = (result["status_code"] >= 400).astype(int)
        np.testing.assert_array_equal(result["is_error"].values, expected.values)

    def test_method_encoding(self, sample_df):
        result = engineer_features(sample_df)
        method_cols = [c for c in result.columns if c.startswith("method_")]
        assert len(method_cols) >= 2

    def test_prepare_single_prediction(self):
        X = prepare_single_prediction("GET", 14, 240.0)
        assert len(X) == 1
        assert X["method_GET"].iloc[0] == 1
        assert X["hour"].iloc[0] == 14

    def test_classification_features_list(self):
        features = get_classification_features()
        assert "hour" in features
        assert "rolling_avg_response" in features
        assert len(features) == 10

    def test_regression_features_list(self):
        features = get_regression_features()
        assert "is_error" in features
        assert len(features) == 11


class TestClassifierPipeline:
    """Tests for the classification model pipeline."""

    @pytest.fixture
    def trained_pipeline(self):
        df = generate_synthetic_dataset(n_records=500, seed=42)
        df = engineer_features(df)
        features = get_classification_features()
        X = df[features]
        y = df["is_error"]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline = ClassifierPipeline()
        pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
        return pipeline, X_test

    def test_trains_all_models(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        assert "logistic_regression" in pipeline.models
        assert "random_forest" in pipeline.models
        assert "xgboost" in pipeline.models

    def test_selects_best_model(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        assert pipeline.best_model_name is not None
        assert pipeline.best_model is not None

    def test_roc_auc_reasonable(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        for name, result in pipeline.results.items():
            assert 0.5 <= result["roc_auc"] <= 1.0, f"{name} ROC-AUC out of range"

    def test_predict_error_probability(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        X = prepare_single_prediction("GET", 14, 240.0)
        result = pipeline.predict_error_probability(X)
        assert 0 <= result["error_probability"] <= 1
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class TestRegressorPipeline:
    """Tests for the regression model pipeline."""

    @pytest.fixture
    def trained_pipeline(self):
        df = generate_synthetic_dataset(n_records=500, seed=42)
        df = engineer_features(df)
        features = get_regression_features()
        X = df[features]
        y = df["response_time_ms"]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = RegressorPipeline()
        pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
        return pipeline, X_test

    def test_trains_all_models(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        assert "linear_regression" in pipeline.models
        assert "random_forest_regressor" in pipeline.models
        assert "gradient_boosting" in pipeline.models

    def test_selects_best_model(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        assert pipeline.best_model_name is not None
        assert pipeline.best_model is not None

    def test_r2_reasonable(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        best_r2 = pipeline.results[pipeline.best_model_name]["r2"]
        assert best_r2 > 0.0, "Best R² should be > 0"

    def test_predict_response_time(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        X = prepare_single_prediction("GET", 14, 240.0)
        X["is_error"] = 0
        reg_features = get_regression_features()
        X = X[reg_features]
        result = pipeline.predict_response_time(X)
        assert result["predicted_response_time_ms"] > 0
        assert result["confidence_interval"]["lower_bound_ms"] >= 0

"""Configuration module for the ML service."""
import os


class Settings:
    """Application settings loaded from environment variables."""

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://logadmin:logadmin123@localhost:5432/logplatform"
    )
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )
    MODELS_DIR: str = os.getenv("MODELS_DIR", "./models")
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    EXPERIMENT_NAME: str = "predictive-log-intelligence"
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2


settings = Settings()

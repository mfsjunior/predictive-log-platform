"""
MLflow Tracker — infrastructure layer for MLflow interaction.
Decouples domain models from MLflow framework.
Includes health check, retry with backoff, and structured error logging.
"""
import time
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MlflowTracker:
    """
    Infrastructure service that handles all MLflow interactions.
    Domain models return results; this tracker logs them to MLflow.
    """

    def __init__(self, tracking_uri: str, experiment_name: str, max_retries: int = 3):
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._max_retries = max_retries
        self._available = False
        self._check_availability()

    def _check_availability(self):
        """Check if MLflow server is reachable."""
        try:
            import mlflow
            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)
            self._available = True
            logger.info(f"MLflow connected: {self._tracking_uri}")
        except Exception as e:
            self._available = False
            logger.warning(f"MLflow not available at {self._tracking_uri}: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    def health_check(self) -> dict:
        """Return MLflow connection status."""
        self._check_availability()
        return {
            "mlflow_available": self._available,
            "tracking_uri": self._tracking_uri,
            "experiment": self._experiment_name,
        }

    def log_classifier_run(self, name: str, model: Any, metrics: dict,
                            is_best: bool = False) -> Optional[str]:
        """Log a classifier model run to MLflow with retry."""
        return self._with_retry(lambda: self._log_run(
            run_name=f"classifier_{name}",
            model=model,
            model_type="classifier",
            model_name=name,
            metrics=metrics,
            is_best=is_best,
        ))

    def log_regressor_run(self, name: str, model: Any, metrics: dict,
                           is_best: bool = False) -> Optional[str]:
        """Log a regressor model run to MLflow with retry."""
        return self._with_retry(lambda: self._log_run(
            run_name=f"regressor_{name}",
            model=model,
            model_type="regressor",
            model_name=name,
            metrics=metrics,
            is_best=is_best,
        ))

    def _log_run(self, run_name: str, model: Any, model_type: str,
                  model_name: str, metrics: dict, is_best: bool) -> Optional[str]:
        """Core MLflow logging logic."""
        if not self._available:
            logger.warning(f"MLflow not available — skipping log for {run_name}")
            return None

        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("is_best", is_best)

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            try:
                if model_name.startswith("xgboost"):
                    import mlflow.xgboost
                    mlflow.xgboost.log_model(model, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception as e:
                logger.warning(f"Model artifact logging failed for {model_name}: {e}")

            return run.info.run_id

    def _with_retry(self, func, retries: int = None) -> Optional[str]:
        """Execute with retry and exponential backoff."""
        retries = retries or self._max_retries
        for attempt in range(retries):
            try:
                return func()
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    f"MLflow attempt {attempt + 1}/{retries} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                if attempt < retries - 1:
                    time.sleep(wait)
                else:
                    logger.error(f"MLflow logging failed after {retries} attempts: {e}")
                    return None

"""
Model Registry — thread-safe singleton for storing trained model instances.
Replaces the global mutable variables in train.py.
"""
import threading
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized, thread-safe registry for trained ML models.
    Singleton pattern — use ModelRegistry.instance() to get the shared instance.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._models: dict[str, Any] = {}
        self._metadata: dict[str, dict] = {}

    @classmethod
    def instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, name: str, model: Any, metadata: Optional[dict] = None):
        """Register a trained model with optional metadata."""
        with self._lock:
            self._models[name] = model
            if metadata:
                self._metadata[name] = metadata
            logger.info(f"Registered model: {name}")

    def get(self, name: str) -> Optional[Any]:
        """Get a registered model by name. Returns None if not found."""
        return self._models.get(name)

    def get_metadata(self, name: str) -> Optional[dict]:
        """Get metadata for a registered model."""
        return self._metadata.get(name)

    def is_loaded(self, name: str) -> bool:
        """Check if a model is registered."""
        return name in self._models

    def status(self) -> dict:
        """Return status of all registered models."""
        return {
            name: {
                "loaded": True,
                "type": type(model).__name__,
                "metadata": self._metadata.get(name, {}),
            }
            for name, model in self._models.items()
        }

    def clear(self):
        """Clear all registered models (for testing)."""
        with self._lock:
            self._models.clear()
            self._metadata.clear()

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

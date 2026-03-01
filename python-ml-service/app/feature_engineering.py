"""
Feature engineering module for web log ML pipeline.
Creates derived features required for classification and regression models.
"""
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to the web log DataFrame.

    Features created:
    - hour: Hour of the day (0-23)
    - day_of_week: Day of the week (0=Monday, 6=Sunday)
    - is_business_hours: Flag for business hours (8-18 weekdays)
    - method_encoded: One-hot encoding of HTTP method
    - rolling_avg_response: Rolling average of response time (window=50)
    - hourly_frequency: Cumulative frequency by hour
    - is_error: Binary target — 1 if status_code >= 400

    Args:
        df: Raw web log DataFrame with at least columns:
            timestamp, method, status_code, response_time_ms

    Returns:
        DataFrame with all original + engineered features.
    """
    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ---------- Temporal Features ----------
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_business_hours"] = (
        (df["hour"] >= 8) & (df["hour"] <= 18) & (df["day_of_week"] <= 4)
    ).astype(int)

    # ---------- HTTP Method Encoding ----------
    method_dummies = pd.get_dummies(df["method"], prefix="method")
    # Ensure all expected methods are present
    for m in ["method_GET", "method_POST", "method_PUT", "method_DELETE", "method_PATCH"]:
        if m not in method_dummies.columns:
            method_dummies[m] = 0
    df = pd.concat([df, method_dummies], axis=1)

    # ---------- Rolling Average Response Time ----------
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["rolling_avg_response"] = (
        df["response_time_ms"]
        .rolling(window=50, min_periods=1)
        .mean()
    )

    # ---------- Hourly Frequency ----------
    hourly_counts = df.groupby("hour").cumcount() + 1
    df["hourly_frequency"] = hourly_counts

    # ---------- Target Variable ----------
    df["is_error"] = (df["status_code"] >= 400).astype(int)

    return df


def get_classification_features() -> list[str]:
    """Return the list of feature column names for classification."""
    return [
        "hour",
        "day_of_week",
        "is_business_hours",
        "method_GET",
        "method_POST",
        "method_PUT",
        "method_DELETE",
        "method_PATCH",
        "rolling_avg_response",
        "hourly_frequency",
    ]


def get_regression_features() -> list[str]:
    """Return the list of feature column names for regression."""
    return [
        "hour",
        "day_of_week",
        "is_business_hours",
        "method_GET",
        "method_POST",
        "method_PUT",
        "method_DELETE",
        "method_PATCH",
        "rolling_avg_response",
        "hourly_frequency",
        "is_error",
    ]


def prepare_single_prediction(
    method: str,
    hour: int,
    historical_avg_response: float,
    day_of_week: int = 2,
) -> pd.DataFrame:
    """
    Prepare a single input record for prediction.

    Args:
        method: HTTP method (GET, POST, etc.)
        hour: Hour of the day (0-23)
        historical_avg_response: Historical average response time in ms
        day_of_week: Day of week (0=Mon, 6=Sun), defaults to Wednesday

    Returns:
        DataFrame with a single row ready for model prediction.
    """
    is_bh = 1 if (8 <= hour <= 18 and day_of_week <= 4) else 0

    data = {
        "hour": [hour],
        "day_of_week": [day_of_week],
        "is_business_hours": [is_bh],
        "method_GET": [1 if method == "GET" else 0],
        "method_POST": [1 if method == "POST" else 0],
        "method_PUT": [1 if method == "PUT" else 0],
        "method_DELETE": [1 if method == "DELETE" else 0],
        "method_PATCH": [1 if method == "PATCH" else 0],
        "rolling_avg_response": [historical_avg_response],
        "hourly_frequency": [50],  # default mid-range
    }

    return pd.DataFrame(data)

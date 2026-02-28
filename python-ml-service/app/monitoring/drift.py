"""
Data drift monitoring using Evidently AI.
Compares current data against training reference to detect statistical drift.
"""
import os
import logging
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Reference data stored in memory after training
_reference_data: pd.DataFrame | None = None


def set_reference_data(df: pd.DataFrame) -> None:
    """Store the training data as reference for drift detection."""
    global _reference_data
    _reference_data = df.copy()
    logger.info(f"Reference data set with {len(df)} samples")


def get_reference_data() -> pd.DataFrame | None:
    """Get the stored reference data."""
    return _reference_data


def generate_drift_report(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame | None = None,
    output_dir: str = "./data",
) -> dict[str, Any]:
    """
    Generate a data drift report comparing current data against reference.

    Uses Evidently AI to compute statistical drift metrics and generate
    an HTML report.

    Args:
        current_data: Current/production DataFrame.
        reference_data: Reference/training DataFrame. Uses stored reference if None.
        output_dir: Directory to save the HTML report.

    Returns:
        Dictionary with drift detection results and report path.
    """
    if reference_data is None:
        reference_data = _reference_data

    if reference_data is None:
        return {
            "error": "No reference data available. Train a model first.",
            "drift_detected": False,
        }

    # Select numeric columns for drift analysis
    numeric_cols = current_data.select_dtypes(include=["number"]).columns.tolist()
    common_cols = [c for c in numeric_cols if c in reference_data.columns]

    if not common_cols:
        return {
            "error": "No common numeric columns between reference and current data.",
            "drift_detected": False,
        }

    ref_subset = reference_data[common_cols].copy()
    curr_subset = current_data[common_cols].copy()

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_subset, current_data=curr_subset)

        # Save HTML report
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")
        report.save_html(report_path)

        # Extract drift results
        report_dict = report.as_dict()

        # Parse drift results from the report
        drift_results = _parse_evidently_report(report_dict, common_cols)
        drift_results["report_path"] = report_path
        drift_results["timestamp"] = timestamp
        drift_results["n_reference_samples"] = len(ref_subset)
        drift_results["n_current_samples"] = len(curr_subset)

        logger.info(f"Drift report generated: {report_path}")
        logger.info(f"Dataset drift detected: {drift_results.get('dataset_drift', False)}")

        return drift_results

    except ImportError:
        logger.warning("Evidently not installed. Using manual drift detection.")
        return _manual_drift_detection(ref_subset, curr_subset, common_cols, output_dir)
    except Exception as e:
        logger.error(f"Drift report generation failed: {e}")
        return _manual_drift_detection(ref_subset, curr_subset, common_cols, output_dir)


def _parse_evidently_report(report_dict: dict, columns: list[str]) -> dict:
    """Parse the Evidently report dictionary to extract drift info."""
    result = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "column_drifts": {},
    }

    try:
        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            metric_result = metric.get("result", {})
            if "drift_share" in metric_result:
                result["dataset_drift"] = metric_result.get("dataset_drift", False)
                result["drift_share"] = metric_result.get("drift_share", 0.0)
                result["n_drifted_columns"] = metric_result.get("number_of_drifted_columns", 0)
                result["n_columns"] = metric_result.get("number_of_columns", 0)
            elif "column_name" in metric_result:
                col_name = metric_result["column_name"]
                result["column_drifts"][col_name] = {
                    "drift_detected": metric_result.get("drift_detected", False),
                    "drift_score": metric_result.get("drift_score", 0.0),
                    "stattest_name": metric_result.get("stattest_name", "unknown"),
                }
    except Exception as e:
        logger.warning(f"Error parsing Evidently report: {e}")

    return result


def _manual_drift_detection(
    ref: pd.DataFrame,
    curr: pd.DataFrame,
    columns: list[str],
    output_dir: str,
) -> dict:
    """Fallback manual drift detection using basic statistical tests."""
    from scipy import stats

    column_drifts = {}
    n_drifted = 0

    for col in columns:
        if col in ref.columns and col in curr.columns:
            ref_vals = ref[col].dropna()
            curr_vals = curr[col].dropna()
            if len(ref_vals) > 0 and len(curr_vals) > 0:
                ks_stat, p_value = stats.ks_2samp(ref_vals, curr_vals)
                drift_detected = p_value < 0.05
                if drift_detected:
                    n_drifted += 1
                column_drifts[col] = {
                    "drift_detected": drift_detected,
                    "drift_score": float(p_value),
                    "stattest_name": "ks_2samp",
                    "ks_statistic": float(ks_stat),
                }

    dataset_drift = n_drifted > len(columns) * 0.5

    return {
        "dataset_drift": dataset_drift,
        "drift_share": n_drifted / max(len(columns), 1),
        "n_drifted_columns": n_drifted,
        "n_columns": len(columns),
        "column_drifts": column_drifts,
        "method": "manual_ks_test",
        "report_path": None,
    }

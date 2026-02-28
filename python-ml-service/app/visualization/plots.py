"""
Visualization module for generating model evaluation plots.
Produces ROC curves, confusion matrices, feature importance, and SHAP values.
"""
import os
import logging

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_roc_curve(
    fpr_dict: dict[str, list],
    tpr_dict: dict[str, list],
    auc_dict: dict[str, float],
    output_path: str,
) -> str:
    """
    Plot ROC curves for multiple classifiers and save as PNG.

    Args:
        fpr_dict: {model_name: fpr_values}
        tpr_dict: {model_name: tpr_values}
        auc_dict: {model_name: auc_value}
        output_path: File path to save the PNG.

    Returns:
        Path to the saved image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, (name, fpr) in enumerate(fpr_dict.items()):
        tpr = tpr_dict[name]
        auc = auc_dict[name]
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC Curve Comparison — Error Classification", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved ROC curve plot to {output_path}")
    return output_path


def plot_confusion_matrix(
    cm: list[list[int]],
    model_name: str,
    output_path: str,
) -> str:
    """
    Plot confusion matrix and save as PNG.

    Args:
        cm: 2D list from sklearn confusion_matrix.
        model_name: Name of the model for the title.
        output_path: File path to save the PNG.

    Returns:
        Path to the saved image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm_array = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm_array, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    labels = ["Normal (2xx/3xx)", "Error (4xx/5xx)"]
    ax.set(
        xticks=np.arange(cm_array.shape[1]),
        yticks=np.arange(cm_array.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=f"Confusion Matrix — {model_name}",
        ylabel="Actual",
        xlabel="Predicted",
    )
    ax.title.set_fontsize(14)
    ax.title.set_fontweight("bold")

    # Text annotations
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(
                j, i, format(cm_array[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_array[i, j] > thresh else "black",
                fontsize=16, fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved confusion matrix plot to {output_path}")
    return output_path


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    model_name: str,
    output_path: str,
) -> str:
    """
    Plot feature importances as horizontal bar chart.

    Args:
        feature_names: List of feature names.
        importances: Array of importance values.
        model_name: Name of the model.
        output_path: File path to save the PNG.

    Returns:
        Path to the saved image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sorted_idx = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_features)))

    ax.barh(range(len(sorted_features)), sorted_importances, color=colors)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=11)
    ax.set_xlabel("Importance", fontsize=14)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=16, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved feature importance plot to {output_path}")
    return output_path


def plot_shap_values(
    model,
    X_sample: pd.DataFrame,
    output_path: str,
) -> str:
    """
    Generate SHAP summary plot for model explainability.

    Args:
        model: Trained model.
        X_sample: Sample of feature data (max 200 rows for performance).
        output_path: File path to save the PNG.

    Returns:
        Path to the saved image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        import shap

        # Limit sample size for performance
        if len(X_sample) > 200:
            X_sample = X_sample.sample(200, random_state=42)

        # Use TreeExplainer for tree-based models, KernelExplainer as fallback
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                X_sample.iloc[:50],
            )
            shap_values = explainer.shap_values(X_sample)

        # Handle multi-output (classification gives list of arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (error)

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False, plot_size=(12, 8))
        plt.title("SHAP Values — Feature Impact on Predictions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved SHAP values plot to {output_path}")

    except Exception as e:
        logger.warning(f"SHAP plot generation failed: {e}. Creating fallback plot.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"SHAP values unavailable:\n{str(e)[:200]}",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.set_title("SHAP Values — Unavailable", fontsize=14)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return output_path

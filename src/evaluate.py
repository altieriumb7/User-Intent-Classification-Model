"""Evaluation utilities for intent classification."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Compute summary metrics, per-class report, and confusion matrix."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose().reset_index(names="label")

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    return metrics, report_frame, matrix_frame


def save_evaluation_artifacts(
    metrics: dict[str, float],
    report_frame: pd.DataFrame,
    matrix_frame: pd.DataFrame,
    metrics_path: Path | str,
    report_path: Path | str,
    matrix_plot_path: Path | str,
) -> None:
    """Persist metrics, per-class report, and a confusion-matrix image."""
    metrics_path = Path(metrics_path)
    report_path = Path(report_path)
    matrix_plot_path = Path(matrix_plot_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_plot_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_frame.to_csv(report_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix_frame.values, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(matrix_frame.columns)),
        yticks=range(len(matrix_frame.index)),
        xticklabels=matrix_frame.columns,
        yticklabels=matrix_frame.index,
        ylabel="True intent",
        xlabel="Predicted intent",
        title="Intent Classification Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix_frame.values.max() / 2 if matrix_frame.values.size else 0
    for i in range(matrix_frame.shape[0]):
        for j in range(matrix_frame.shape[1]):
            value = matrix_frame.iloc[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(matrix_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

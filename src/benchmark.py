"""Cross-validation benchmark utilities for model robustness checks."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate

from src.model import build_baseline_pipeline
from src.preprocessing import clean_text


def run_cross_validation_report(
    texts: list[str],
    labels: list[int],
    folds: int = 5,
    random_state: int = 42,
) -> dict[str, float | int]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    pipeline = build_baseline_pipeline()
    cleaned = [clean_text(text) for text in texts]
    scores = cross_validate(
        pipeline,
        cleaned,
        labels,
        scoring=("accuracy", "f1_macro", "f1_micro"),
        cv=cv,
        n_jobs=None,
    )
    return {
        "folds": folds,
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "macro_f1_mean": float(scores["test_f1_macro"].mean()),
        "macro_f1_std": float(scores["test_f1_macro"].std()),
        "micro_f1_mean": float(scores["test_f1_micro"].mean()),
        "micro_f1_std": float(scores["test_f1_micro"].std()),
    }


def save_cross_validation_report(report: dict[str, float | int], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

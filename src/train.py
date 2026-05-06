"""Train and evaluate the baseline intent classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    DEMO_DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    ROUTING_REPORT_PATH,
    MONITORING_REPORT_PATH,
    CV_REPORT_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from src.data import load_dataset, split_dataset
from src.evaluate import evaluate_predictions, save_evaluation_artifacts
from src.model import IntentModel, save_model, train_baseline
from src.routing import simulate_routing
from src.benchmark import run_cross_validation_report, save_cross_validation_report
from src.monitoring import build_monitoring_report, save_monitoring_report


def infer_synthetic_flag(data) -> bool:
    """Return True only when a dataset explicitly marks every row as synthetic."""
    if "is_synthetic" not in data.columns:
        return False
    values = data["is_synthetic"].astype(str).str.lower().str.strip()
    return bool(values.isin({"true", "1", "yes"}).all())


def train_and_evaluate(
    data_path: Path = DEMO_DATA_PATH,
    model_path: Path = MODEL_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, object]:
    """Train the baseline model, save artifacts, and return evaluation results."""
    data = load_dataset(data_path)
    splits = split_dataset(data)

    pipeline = train_baseline(
        splits.train["clean_message"].tolist(),
        splits.train["label"].tolist(),
    )
    intent_model = IntentModel(pipeline=pipeline, label_encoder=splits.label_encoder)
    save_model(intent_model, model_path)

    validation_predictions = intent_model.predict(splits.validation["clean_message"].tolist())
    test_predictions = intent_model.predict(splits.test["clean_message"].tolist())
    labels = intent_model.labels

    metrics, report_frame, matrix_frame = evaluate_predictions(
        splits.test["intent"].tolist(),
        test_predictions,
        labels,
    )
    save_evaluation_artifacts(
        metrics,
        report_frame,
        matrix_frame,
        reports_dir / METRICS_PATH.name,
        reports_dir / CLASSIFICATION_REPORT_PATH.name,
        reports_dir / CONFUSION_MATRIX_PATH.name,
    )


    cv_report = run_cross_validation_report(
        splits.train["clean_message"].tolist(),
        splits.train["label"].tolist(),
    )
    save_cross_validation_report(cv_report, reports_dir / CV_REPORT_PATH.name)

    test_probabilities = intent_model.predict_proba(splits.test["clean_message"].tolist())
    test_confidences = test_probabilities.max(axis=1).tolist()
    monitoring_report = build_monitoring_report(
        test_predictions,
        test_confidences,
        low_confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    save_monitoring_report(monitoring_report, reports_dir / MONITORING_REPORT_PATH.name)

    routing_report = simulate_routing(splits.test["intent"].tolist(), test_predictions)
    routing_path = reports_dir / ROUTING_REPORT_PATH.name
    routing_path.parent.mkdir(parents=True, exist_ok=True)
    routing_path.write_text(json.dumps(routing_report, indent=2), encoding="utf-8")

    return {
        "data_path": str(data_path),
        "model_path": str(model_path),
        "metrics": metrics,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "routing_report": routing_report,
        "cross_validation_report": cv_report,
        "monitoring_report": monitoring_report,
        "synthetic_data": infer_synthetic_flag(data),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TF-IDF Logistic Regression intent model.")
    parser.add_argument("--data", type=Path, default=DEMO_DATA_PATH, help="CSV with message and intent columns.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Where to save the trained model.")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR, help="Where to save evaluation artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_evaluate(args.data, args.model_path, args.reports_dir)
    print("Training complete")
    print(f"Dataset: {result['data_path']}")
    print(f"Synthetic data: {result['synthetic_data']}")
    print(f"Model saved to: {result['model_path']}")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Misrouting rate: {result['routing_report']['misrouting_rate']:.3f}")


if __name__ == "__main__":
    main()

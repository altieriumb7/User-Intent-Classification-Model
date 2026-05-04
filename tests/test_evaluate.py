from src.evaluate import evaluate_predictions


def test_evaluate_predictions_includes_required_metrics_and_per_class_rows():
    labels = ["billing", "refund"]
    metrics, report, matrix = evaluate_predictions(
        ["billing", "billing", "refund"],
        ["billing", "refund", "refund"],
        labels,
    )

    assert set(metrics) == {"accuracy", "macro_f1", "micro_f1"}
    assert metrics["accuracy"] == 2 / 3
    assert set(labels).issubset(set(report["label"]))
    assert matrix.shape == (2, 2)

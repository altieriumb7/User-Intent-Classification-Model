import pandas as pd

from src.train import infer_synthetic_flag, train_and_evaluate


def test_infer_synthetic_flag_handles_demo_and_external_data():
    assert infer_synthetic_flag(pd.DataFrame({"is_synthetic": [True, "true", "1"]}))
    assert not infer_synthetic_flag(pd.DataFrame({"message": ["hello"], "intent": ["billing"]}))


def test_train_and_evaluate_writes_all_artifacts(tmp_path):
    model_path = tmp_path / "intent_classifier.joblib"
    reports_dir = tmp_path / "reports"

    result = train_and_evaluate(model_path=model_path, reports_dir=reports_dir)

    assert model_path.exists()
    for name in [
        "evaluation_metrics.json",
        "classification_report.csv",
        "confusion_matrix.png",
        "routing_report.json",
        "cross_validation_report.json",
        "monitoring_report.json",
    ]:
        assert (reports_dir / name).exists()
    assert result["synthetic_data"] is True

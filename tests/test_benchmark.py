from src.data import load_dataset, split_dataset
from src.benchmark import run_cross_validation_report


def test_cross_validation_report_has_expected_keys():
    data = load_dataset()
    splits = split_dataset(data)
    report = run_cross_validation_report(
        splits.train["clean_message"].tolist(),
        splits.train["label"].tolist(),
        folds=3,
    )
    assert report["folds"] == 3
    assert 0 <= report["accuracy_mean"] <= 1
    assert 0 <= report["macro_f1_mean"] <= 1

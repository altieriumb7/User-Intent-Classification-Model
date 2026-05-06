from src.monitoring import build_monitoring_report


def test_monitoring_report_counts_low_confidence_and_distribution():
    report = build_monitoring_report(
        predicted_intents=["billing", "refund", "billing"],
        confidences=[0.9, 0.3, 0.4],
        low_confidence_threshold=0.5,
    )
    assert report["total_predictions"] == 3
    assert report["low_confidence_count"] == 2
    assert report["intent_distribution"]["billing"] == 2

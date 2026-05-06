"""Monitoring helpers for production-style drift and confidence tracking."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def build_monitoring_report(
    predicted_intents: list[str],
    confidences: list[float],
    low_confidence_threshold: float,
) -> dict[str, object]:
    if len(predicted_intents) != len(confidences):
        raise ValueError("predicted_intents and confidences must have the same length.")

    total = len(predicted_intents)
    low_confidence = [c for c in confidences if c < low_confidence_threshold]
    intent_distribution = Counter(predicted_intents)

    return {
        "total_predictions": total,
        "low_confidence_threshold": low_confidence_threshold,
        "low_confidence_count": len(low_confidence),
        "low_confidence_rate": (len(low_confidence) / total) if total else 0.0,
        "avg_confidence": (sum(confidences) / total) if total else 0.0,
        "intent_distribution": dict(intent_distribution),
    }


def save_monitoring_report(report: dict[str, object], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

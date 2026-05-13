"""Command-line inference for the intent classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import DEFAULT_CONFIDENCE_THRESHOLD, MODEL_PATH
from src.model import IntentModel, load_model
from src.routing import route_intent


TRIAGE_INTENT = "needs_human_review"
TRIAGE_TEAM = "Manual Triage Desk"


def predict_with_model(
    text: str,
    model: IntentModel,
    top_k: int = 3,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, object]:
    """Predict intent and team with a preloaded model."""
    prediction = model.predict_one(text, top_k=top_k)
    confidence = float(prediction["confidence"])
    is_low_confidence = confidence < confidence_threshold

    prediction["suggested_intent"] = prediction["intent"]
    prediction["needs_human_review"] = is_low_confidence
    prediction["confidence_threshold"] = confidence_threshold

    if is_low_confidence:
        prediction["intent"] = TRIAGE_INTENT
        prediction["support_team"] = TRIAGE_TEAM
    else:
        prediction["support_team"] = route_intent(str(prediction["intent"]))
    return prediction


def predict_text(
    text: str,
    model_path: Path = MODEL_PATH,
    top_k: int = 3,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, object]:
    """Predict intent and team, with low-confidence fallback to manual triage."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python -m src.train` first."
        )
    return predict_with_model(text, load_model(model_path), top_k, confidence_threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict support-ticket intent from text.")
    parser.add_argument("--text", required=True, help="Support message or ticket text.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Path to trained joblib model.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of ranked intents to return.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Fallback to manual triage when confidence is below this threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_text(args.text, args.model_path, args.top_k, args.confidence_threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

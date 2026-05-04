"""Command-line inference for the intent classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import MODEL_PATH
from src.model import load_model
from src.routing import route_intent


def predict_text(text: str, model_path: Path = MODEL_PATH, top_k: int = 3) -> dict[str, object]:
    """Predict intent, confidence, top-k probabilities, and routing for one message."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python -m src.train` first."
        )
    model = load_model(model_path)
    prediction = model.predict_one(text, top_k=top_k)
    prediction["support_team"] = route_intent(str(prediction["intent"]))
    return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict support-ticket intent from text.")
    parser.add_argument("--text", required=True, help="Support message or ticket text.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Path to trained joblib model.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of ranked intents to return.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_text(args.text, args.model_path, args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

from src.config import MODELS_DIR
from src.data import load_dataset, split_dataset
from src.model import IntentModel, save_model, train_baseline
from src.predict_intent import TRIAGE_INTENT, TRIAGE_TEAM, predict_text
from src.routing import INTENT_TO_TEAM


def test_predict_text_returns_intent_confidence_top_predictions_and_team():
    data = load_dataset()
    splits = split_dataset(data)
    pipeline = train_baseline(
        splits.train["clean_message"].tolist(),
        splits.train["label"].tolist(),
    )
    model = IntentModel(pipeline=pipeline, label_encoder=splits.label_encoder)
    model_path = MODELS_DIR / "test_intent_classifier.joblib"

    try:
        save_model(model, model_path)
        result = predict_text("I cannot access my account", model_path=model_path)
    finally:
        if model_path.exists():
            model_path.unlink()

    assert result["suggested_intent"] in model.labels
    assert result["intent"] in (set(model.labels) | {TRIAGE_INTENT})
    assert 0 <= result["confidence"] <= 1
    assert len(result["top_predictions"]) == 3
    assert result["support_team"] in (set(INTENT_TO_TEAM.values()) | {TRIAGE_TEAM})


def test_predict_text_routes_to_manual_triage_when_confidence_is_below_threshold():
    data = load_dataset()
    splits = split_dataset(data)
    pipeline = train_baseline(
        splits.train["clean_message"].tolist(),
        splits.train["label"].tolist(),
    )
    model = IntentModel(pipeline=pipeline, label_encoder=splits.label_encoder)
    model_path = MODELS_DIR / "test_intent_classifier.joblib"

    try:
        save_model(model, model_path)
        result = predict_text("help", model_path=model_path, confidence_threshold=0.99)
    finally:
        if model_path.exists():
            model_path.unlink()

    assert result["needs_human_review"] is True
    assert result["intent"] == TRIAGE_INTENT
    assert result["support_team"] == TRIAGE_TEAM

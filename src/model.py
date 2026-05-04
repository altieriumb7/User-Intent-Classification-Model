"""Baseline intent classifier training and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import clean_text


@dataclass
class IntentModel:
    """Serializable wrapper for a trained classifier and label encoder."""

    pipeline: Pipeline
    label_encoder: LabelEncoder

    @property
    def labels(self) -> list[str]:
        return self.label_encoder.classes_.tolist()

    def predict(self, texts: list[str]) -> list[str]:
        encoded = self.pipeline.predict([clean_text(text) for text in texts])
        return self.label_encoder.inverse_transform(encoded).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba([clean_text(text) for text in texts])

    def predict_one(self, text: str, top_k: int = 3) -> dict[str, object]:
        probabilities = self.predict_proba([text])[0]
        order = np.argsort(probabilities)[::-1]
        top_indices = order[:top_k]
        top_predictions = [
            {
                "intent": self.label_encoder.inverse_transform([idx])[0],
                "probability": float(probabilities[idx]),
            }
            for idx in top_indices
        ]
        return {
            "intent": top_predictions[0]["intent"],
            "confidence": top_predictions[0]["probability"],
            "top_predictions": top_predictions,
        }


def build_baseline_pipeline() -> Pipeline:
    """Build the TF-IDF + Logistic Regression baseline classifier."""
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                lowercase=False,
                                ngram_range=(1, 3),
                                min_df=1,
                                max_df=0.95,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                lowercase=False,
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=1,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def train_baseline(texts: list[str], labels: list[int]) -> Pipeline:
    """Fit the baseline model on cleaned ticket text and encoded labels."""
    pipeline = build_baseline_pipeline()
    pipeline.fit([clean_text(text) for text in texts], labels)
    return pipeline


def save_model(model: IntentModel, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path | str) -> IntentModel:
    return joblib.load(path)

"""Dataset loading, validation, and splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import DEMO_DATA_PATH, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE
from src.preprocessing import clean_text


REQUIRED_COLUMNS = {"message", "intent"}


@dataclass(frozen=True)
class DatasetSplits:
    """Container for train, validation, and test data."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    label_encoder: LabelEncoder


def load_dataset(path: Path | str = DEMO_DATA_PATH) -> pd.DataFrame:
    """Load a support-ticket dataset and add a cleaned text column."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. This project ships with demo data at "
            f"{DEMO_DATA_PATH}."
        )

    data = pd.read_csv(dataset_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data = data.dropna(subset=["message", "intent"]).copy()
    data["message"] = data["message"].astype(str)
    data["intent"] = data["intent"].astype(str).str.strip()
    data["clean_message"] = data["message"].map(clean_text)
    data = data[data["clean_message"].str.len() > 0].reset_index(drop=True)
    return data


def split_dataset(
    data: pd.DataFrame,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DatasetSplits:
    """Create stratified train, validation, and test splits with encoded labels."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1.")

    label_counts = data["intent"].value_counts()
    if (label_counts < 3).any():
        sparse = label_counts[label_counts < 3].index.tolist()
        raise ValueError(f"Each intent needs at least 3 examples for stratified splitting: {sparse}")

    label_encoder = LabelEncoder()
    data = data.copy()
    data["label"] = label_encoder.fit_transform(data["intent"])

    holdout_size = validation_size + test_size
    train, holdout = train_test_split(
        data,
        test_size=holdout_size,
        stratify=data["intent"],
        random_state=random_state,
    )

    relative_test_size = test_size / holdout_size
    validation, test = train_test_split(
        holdout,
        test_size=relative_test_size,
        stratify=holdout["intent"],
        random_state=random_state,
    )

    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        label_encoder=label_encoder,
    )

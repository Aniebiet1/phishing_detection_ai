"""Model loading and prediction helpers."""

from __future__ import annotations

import os
from typing import Any, Tuple

import joblib


def load_model(path: str) -> Any:
    """Load a persisted model from disk.

    Args:
        path: Path to a joblib-saved model.

    Returns:
        The loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: Any exception raised by joblib.load.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Please train a model and save it to this location."
        )

    return joblib.load(path)


def predict_text(model: Any, text: str) -> Tuple[str, float]:
    """Make a prediction for a single text input.

    Args:
        model: A scikit-learn-like model with `predict` and optionally `predict_proba`.
        text: The raw text / URL to classify.

    Returns:
        A tuple of (label, score).
    """

    raw_label = model.predict([text])[0]

    score = 1.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])
        # Use the max probability across classes as the confidence score.
        score = float(max(proba[0]))

    label_str = str(raw_label).strip().lower()
    if label_str in {"1", "true", "phishing", "spam", "malicious", "fraud"}:
        normalized_label = "phishing"
    elif label_str in {"0", "false", "legitimate", "ham", "benign", "safe"}:
        normalized_label = "legitimate"
    else:
        normalized_label = str(raw_label)

    return normalized_label, float(score)

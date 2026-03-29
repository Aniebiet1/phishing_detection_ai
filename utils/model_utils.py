"""Model loading and prediction helpers."""

from __future__ import annotations

import math
import os
import re
import sys
from urllib.parse import urlparse
from typing import Any, Tuple

import joblib

from utils.vectorizer_utils import truncate_for_char_features


def _register_legacy_pickle_symbols() -> None:
    """Expose legacy symbol names used by older serialized pipelines.

    Older models were trained with helper functions in script scope, so pickle stored
    references like `__main__._truncate_for_char_features`. During API startup under
    uvicorn, `__main__` points to uvicorn's launcher module, so we attach the helper
    there (and on `__mp_main__` if present) for backward-compatible loading.
    """

    for module_name in ("__main__", "__mp_main__"):
        module = sys.modules.get(module_name)
        if module is not None and not hasattr(module, "_truncate_for_char_features"):
            setattr(module, "_truncate_for_char_features", truncate_for_char_features)


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

    _register_legacy_pickle_symbols()
    return joblib.load(path)


LEETSPEAK_MAP = str.maketrans(
    {
        "0": "o",
        "1": "l",
        "2": "z",
        "3": "e",
        "4": "a",
        "5": "s",
        "6": "g",
        "7": "t",
        "8": "b",
        "9": "g",
    }
)

BRAND_TOKENS = {
    "google",
    "paypal",
    "apple",
    "microsoft",
    "amazon",
    "netflix",
    "facebook",
    "instagram",
    "chase",
    "bankofamerica",
}


def _extract_hostname(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return ""

    parsed = urlparse(candidate)
    if parsed.hostname:
        return parsed.hostname.lower()

    parsed = urlparse(f"http://{candidate}")
    return (parsed.hostname or "").lower()


def _looks_like_url(text: str) -> bool:
    hostname = _extract_hostname(text)
    return bool(hostname and "." in hostname)


def _is_typosquatting_like_url(text: str) -> bool:
    hostname = _extract_hostname(text)
    if not hostname:
        return False

    left_most_label = hostname.split(".")[0]
    if not any(ch.isdigit() for ch in left_most_label):
        return False

    normalized = re.sub(r"[^a-z0-9]", "", left_most_label).translate(LEETSPEAK_MAP)
    return any(brand in normalized for brand in BRAND_TOKENS)


def _normalize_raw_label(raw_label: Any) -> str:
    label_str = str(raw_label).strip().lower()
    if label_str in {"1", "true", "phishing", "spam", "malicious", "fraud"}:
        return "phishing"
    if label_str in {"0", "false", "legitimate", "ham", "benign", "safe"}:
        return "legitimate"
    return str(raw_label)


def _get_phishing_probability(model: Any, text: str) -> float | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        classes = list(getattr(model, "classes_", []))
        if classes and 1 in classes:
            idx = classes.index(1)
            return float(proba[idx])
        if len(proba) >= 2:
            return float(proba[1])
        return float(proba[0])

    if hasattr(model, "decision_function"):
        decision = model.decision_function([text])
        try:
            margin = float(decision[0])  # type: ignore[index]
        except Exception:
            return None
        return float(1.0 / (1.0 + math.exp(-margin)))

    return None


def predict_text_with_metadata(model: Any, text: str, source: str | None = None) -> tuple[str, float, dict[str, Any]]:
    """Predict label and score with URL-aware heuristics and debugging metadata."""

    raw_label = model.predict([text])[0]
    normalized_label = _normalize_raw_label(raw_label)

    phishing_probability = _get_phishing_probability(model, text)
    if phishing_probability is not None:
        score = float(max(phishing_probability, 1.0 - phishing_probability))
    else:
        score = 0.5

    inferred_url = _looks_like_url(text)
    source_is_url = str(source or "").strip().lower() == "url"
    threshold = 0.50

    if phishing_probability is not None:
        normalized_label = "phishing" if phishing_probability >= threshold else "legitimate"

    typosquatting_like = _is_typosquatting_like_url(text)
    if typosquatting_like and normalized_label == "legitimate":
        normalized_label = "phishing"
        score = max(score, 0.70)

    details: dict[str, Any] = {
        "score_type": "confidence_not_accuracy",
        "decision_threshold": float(threshold),
        "inferred_url": bool(inferred_url),
        "typosquatting_like": bool(typosquatting_like),
    }
    if phishing_probability is not None:
        details["phishing_probability"] = float(phishing_probability)

    return normalized_label, float(score), details


def predict_text(model: Any, text: str) -> Tuple[str, float]:
    """Make a prediction for a single text input.

    Args:
        model: A scikit-learn-like model with `predict` and optionally `predict_proba`.
        text: The raw text / URL to classify.

    Returns:
        A tuple of (label, score).
    """

    label, score, _ = predict_text_with_metadata(model, text=text, source=None)
    return label, score

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DEFAULT_MODELS = [
    "logistic_regression",
    "linear_svc",
    "multinomial_nb",
    "sgd_classifier",
    "passive_aggressive",
]
SELECTION_METRICS = ["phishing_f1", "accuracy", "macro_f1", "weighted_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a phishing detection text model using email and URL datasets."
    )
    parser.add_argument(
        "--email-data",
        type=str,
        default="data/phishing_email.csv",
        help="Path to email dataset CSV (must contain text_combined and label columns).",
    )
    parser.add_argument(
        "--url-data",
        type=str,
        default="data/PhiUSIIL_Phishing_URL_Dataset.csv",
        help="Path to URL dataset CSV (must contain URL and label columns).",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="saved_models/phishing_model.joblib",
        help="Output path for trained model.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default="saved_models/training_report.json",
        help="Output path for training report.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="phishing_f1",
        choices=SELECTION_METRICS,
        help="Metric used to rank models and select the best one.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=(
            "Comma-separated model names to evaluate. "
            "Available: logistic_regression, linear_svc, multinomial_nb, sgd_classifier, passive_aggressive"
        ),
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=100_000,
        help="Maximum TF-IDF vocabulary size. Lower this to use less memory (default: 100000).",
    )
    return parser.parse_args()


def normalize_binary_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        labels = series.fillna(0).astype(int)
        if not set(labels.unique()).issubset({0, 1}):
            raise ValueError("Numeric labels must be binary (0/1).")
        return labels

    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "ham": 0,
        "legitimate": 0,
        "benign": 0,
        "safe": 0,
        "spam": 1,
        "phishing": 1,
        "malicious": 1,
        "fraud": 1,
    }

    normalized = series.fillna("").astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        unknown = sorted(series[normalized.isna()].astype(str).str.strip().unique().tolist())
        raise ValueError(f"Unrecognized label values found: {unknown[:10]}")
    return normalized.astype(int)


def load_email_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"text_combined", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Email dataset missing columns: {sorted(missing)}")

    out = df[["text_combined", "label"]].copy()
    out = out.rename(columns={"text_combined": "text"})
    out["source"] = "email"
    return out


def load_url_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"URL", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"URL dataset missing columns: {sorted(missing)}")

    out = df[["URL", "label"]].copy()
    out = out.rename(columns={"URL": "text"})
    out["source"] = "url"
    return out


def prepare_training_data(email_path: Path, url_path: Path) -> pd.DataFrame:
    email_df = load_email_dataset(email_path)
    url_df = load_url_dataset(url_path)

    combined = pd.concat([email_df, url_df], axis=0, ignore_index=True)
    combined["text"] = combined["text"].fillna("").astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0].copy()
    combined["label"] = normalize_binary_labels(combined["label"])

    return combined


def build_vectorizer(max_features: int = 100_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        max_features=max_features,
    )


def get_model_candidates(random_state: int, max_features: int = 100_000) -> dict[str, Callable[[], BaseEstimator]]:
    return {
        "logistic_regression": lambda: Pipeline(
            steps=[
                ("tfidf", build_vectorizer(max_features=max_features)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svc": lambda: Pipeline(
            steps=[
                ("tfidf", build_vectorizer(max_features=max_features)),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "multinomial_nb": lambda: Pipeline(
            steps=[
                ("tfidf", build_vectorizer(max_features=max_features)),
                ("clf", MultinomialNB(alpha=0.5)),
            ]
        ),
        "sgd_classifier": lambda: Pipeline(
            steps=[
                ("tfidf", build_vectorizer(max_features=max_features)),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=1e-5,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "passive_aggressive": lambda: Pipeline(
            steps=[
                ("tfidf", build_vectorizer(max_features=max_features)),
                (
                    "clf",
                    PassiveAggressiveClassifier(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def parse_requested_models(models_arg: str, available: dict[str, Any]) -> list[str]:
    requested = [m.strip() for m in models_arg.split(",") if m.strip()]
    if not requested:
        raise ValueError("No models requested. Provide at least one model name via --models.")

    unknown = [m for m in requested if m not in available]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}. Available: {sorted(available.keys())}")

    return requested


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    precision, recall, phishing_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    matrix = confusion_matrix(y_true, y_pred).tolist()
    full_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision_phishing": float(precision),
        "recall_phishing": float(recall),
        "phishing_f1": float(phishing_f1),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": matrix,
        "classification_report": full_report,
    }


def train_and_evaluate_models(
    x_train: pd.Series,
    y_train: pd.Series,
    x_test: pd.Series,
    y_test: pd.Series,
    model_builders: dict[str, Callable[[], BaseEstimator]],
    requested_models: list[str],
) -> tuple[list[dict[str, Any]], dict[str, BaseEstimator]]:
    results: list[dict[str, Any]] = []
    fitted_models: dict[str, BaseEstimator] = {}

    for model_name in requested_models:
        model = model_builders[model_name]()

        train_start = perf_counter()
        model.fit(x_train, y_train)
        train_seconds = perf_counter() - train_start

        predict_start = perf_counter()
        y_pred = model.predict(x_test)
        predict_seconds = perf_counter() - predict_start

        metrics = evaluate_predictions(y_test, y_pred)
        metrics["name"] = model_name
        metrics["train_seconds"] = float(train_seconds)
        metrics["predict_seconds"] = float(predict_seconds)

        results.append(metrics)
        fitted_models[model_name] = model

    return results, fitted_models


def rank_models(results: list[dict[str, Any]], selection_metric: str) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda item: (
            item[selection_metric],
            item["macro_f1"],
            item["accuracy"],
        ),
        reverse=True,
    )


def print_ranking_table(ranked_results: list[dict[str, Any]], selection_metric: str) -> None:
    print("\nModel Ranking")
    print("=" * 72)
    print(
        f"{'Model':<22} {'Accuracy':>10} {'Phish_F1':>10} {'Macro_F1':>10} "
        f"{selection_metric:>12}"
    )
    print("-" * 72)

    for item in ranked_results:
        print(
            f"{item['name']:<22} {item['accuracy']:>10.4f} {item['phishing_f1']:>10.4f} "
            f"{item['macro_f1']:>10.4f} {item[selection_metric]:>12.4f}"
        )


def main() -> None:
    args = parse_args()

    email_path = Path(args.email_data)
    url_path = Path(args.url_data)
    model_out = Path(args.model_out)
    report_out = Path(args.report_out)

    if not email_path.exists():
        raise FileNotFoundError(f"Email dataset not found: {email_path}")
    if not url_path.exists():
        raise FileNotFoundError(f"URL dataset not found: {url_path}")

    if args.selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"Invalid selection metric '{args.selection_metric}'. "
            f"Choose from: {SELECTION_METRICS}"
        )

    df = prepare_training_data(email_path=email_path, url_path=url_path)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    model_builders = get_model_candidates(random_state=args.random_state, max_features=args.max_features)
    requested_models = parse_requested_models(args.models, model_builders)

    results, fitted_models = train_and_evaluate_models(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        model_builders=model_builders,
        requested_models=requested_models,
    )
    ranked_results = rank_models(results, selection_metric=args.selection_metric)

    best_model_name = ranked_results[0]["name"]
    best_model = fitted_models[best_model_name]

    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_out)

    training_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "email_dataset": str(email_path),
            "url_dataset": str(url_path),
            "rows_used": int(len(df)),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "label_distribution": {
                "legitimate_0": int((df["label"] == 0).sum()),
                "phishing_1": int((df["label"] == 1).sum()),
            },
        },
        "selection": {
            "metric": args.selection_metric,
            "best_model": best_model_name,
        },
        "models": ranked_results,
        "ranking": [result["name"] for result in ranked_results],
        "model_output": str(model_out),
    }

    with report_out.open("w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    print_ranking_table(ranked_results, selection_metric=args.selection_metric)
    print("\nTraining complete.")
    print(f"Best model: {best_model_name}")
    print(f"Best {args.selection_metric}: {ranked_results[0][args.selection_metric]:.4f}")
    print(f"Model saved to: {model_out}")
    print(f"Report saved to: {report_out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gc
import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from utils.vectorizer_utils import truncate_for_char_features


DEFAULT_MODELS = [
    "logistic_regression",
    "linear_svc",
    "calibrated_linear_svc",
    "multinomial_nb",
    "sgd_classifier",
    "passive_aggressive",
]
SELECTION_METRICS = ["phishing_f1", "accuracy", "macro_f1", "weighted_f1"]

PHISHING_URL_HARD_EXAMPLES = [
    "amnazon.com",
    "paypa1.com",
    "goggle-login.net",
    "faceboook-security.com",
    "waImart.com",
    "appIe-id.com",
    "rnicrosoft.com",
    "netflix-billing.co",
    "instagrarn.com",
    "wellsfarg0.com",
    "chase-bank.net",
    "banco-itauu.com",
    "bankoffamerica.com",
    "vvernmo.com",
    "twitter-verify.org",
    "linnkedin.com",
    "fedexpress.com",
    "us-ps.com",
    "paypals-support.com",
    "microsoft-365-update.com",
    "paypal.com.secure-login.ru",
    "amazon.prime-verify.it",
    "apple.id-unlock.com.cn",
    "microsoft.online-office.weebly.com",
    "chase.account-update.xyz",
    "netflix.payment-fail.site",
    "google.security-apps.com.br",
    "facebook.login-recovery.online",
    "bankofamerica.com-access.tk",
    "wellsfargo.verification.herokuapp.com",
    "irs.gov.payment-tax.net",
    "fedex.track-package.info",
    "ups.delivery-notice.co",
    "coinbase.secure-wallet.sh",
    "binance.account-verify.app",
    "steam.gift-cards.discount",
    "roblox.robux-free.pro",
    "disneyplus.subscription-renewal.com",
    "adobe.pdf-reader-update.com",
    "zoom.meeting-invite.us",
    "your-account-has-been-locked-verify-now.com",
    "secure-banking-alert-update.net",
    "unauthorized-login-attempt-detection.org",
    "confirm-your-identity-immediately.com",
    "billing-issue-resolved-login.com",
    "urgent-security-notification.net",
    "tax-refund-portal-claim-now.gov.site",
    "missing-package-redelivery-form.com",
    "social-security-administration-alert.com",
    "payroll-bonus-distribution-2026.com",
    "it-helpdesk-password-reset-request.net",
    "legal-notice-against-your-account.com",
    "suspicious-activity-detected-login.com",
    "failed-payment-notification-update.com",
    "verify-your-credentials-to-avoid-suspension.com",
    "refund-request-processing-center.com",
    "loyalty-points-expiring-claim-now.com",
    "exclusive-offer-valid-for-24-hours.net",
    "membership-cancellation-avoidance.com",
    "secure-web-portal-access.com",
    "bit.ly/secure-paypal-login",
    "tinyurl.com/security-update",
    "t.co/XyZ123-verify",
    "cutt.ly/bank-security",
    "short.io/irs-payment",
]

LEGITIMATE_URL_HARD_NEGATIVES = [
    "google.com",
    "www.google.com",
    "accounts.google.com",
    "amazon.com",
    "www.amazon.com",
    "paypal.com",
    "www.paypal.com",
    "apple.com",
    "appleid.apple.com",
    "microsoft.com",
    "login.microsoftonline.com",
    "facebook.com",
    "instagram.com",
    "netflix.com",
    "wellsfargo.com",
    "chase.com",
    "linkedin.com",
    "venmo.com",
    "bankofamerica.com",
    "fedex.com",
    "usps.com",
    "ups.com",
    "irs.gov",
    "adobe.com",
    "zoom.us",
    "steamcommunity.com",
    "help.netflix.com",
    "support.google.com",
    "status.aws.amazon.com",
    "docs.python.org",
]

COMMON_BRANDS = [
    "google",
    "paypal",
    "amazon",
    "facebook",
    "apple",
    "microsoft",
    "netflix",
    "instagram",
    "wellsfargo",
    "linkedin",
    "venmo",
]


def _generate_typosquatting_variants() -> list[str]:
    replacements = {
        "a": "4",
        "e": "3",
        "i": "1",
        "o": "0",
        "l": "1",
        "m": "rn",
        "w": "vv",
    }
    suffixes = ["-secure.com", "-verify.net", "-login.org", "-support.com"]
    variants: set[str] = set()

    for brand in COMMON_BRANDS:
        lowered = brand.lower()
        for ch, repl in replacements.items():
            if ch in lowered:
                variants.add(lowered.replace(ch, repl, 1) + ".com")

        if len(lowered) > 3:
            variants.add(lowered[:3] + lowered[3] + lowered[3:] + ".com")

        variants.add(lowered + "s" + ".com")
        for suffix in suffixes:
            variants.add(lowered + suffix)

    return sorted(variants)


def _generate_brand_in_subdomain_spoofs() -> list[str]:
    patterns = [
        "{brand}.security-check-login.ru",
        "{brand}.account-verify-session.cn",
        "{brand}.billing-review-update.site",
        "{brand}.support-center-reset.online",
    ]
    return [pattern.format(brand=brand) for brand in COMMON_BRANDS for pattern in patterns]


def _generate_urgent_keyword_domains() -> list[str]:
    phrases = [
        "verify-account-now",
        "failed-payment-alert",
        "security-check-required",
        "identity-confirmation-center",
        "unauthorized-login-warning",
        "suspended-access-unlock",
    ]
    tlds = [".com", ".net", ".org"]
    out: list[str] = []
    for phrase in phrases:
        for tld in tlds:
            out.append(f"{phrase}{tld}")
            out.append(f"{phrase}-2026{tld}")
    return out


def _build_augmented_url_dataset(repeat: int = 1) -> pd.DataFrame:
    if repeat < 1:
        raise ValueError("--url-augmentation-weight must be >= 1")

    phishing_urls = sorted(
        set(
            PHISHING_URL_HARD_EXAMPLES
            + _generate_typosquatting_variants()
            + _generate_brand_in_subdomain_spoofs()
            + _generate_urgent_keyword_domains()
        )
    )

    phishing_rows = [
        {"text": url, "label": 1, "source": "url_augmented"}
        for _ in range(repeat)
        for url in phishing_urls
    ]
    legitimate_rows = [
        {"text": url, "label": 0, "source": "url_augmented"}
        for _ in range(repeat)
        for url in LEGITIMATE_URL_HARD_NEGATIVES
    ]

    augmented = pd.DataFrame(phishing_rows + legitimate_rows)
    augmented["text"] = augmented["text"].astype(str).str.strip().str.lower()
    augmented = augmented.drop_duplicates(subset=["text", "label"])
    return augmented


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
        default="accuracy",
        choices=SELECTION_METRICS,
        help="Metric used to rank models and select the best one.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=(
            "Comma-separated model names to evaluate. "
            "Available: logistic_regression, linear_svc, calibrated_linear_svc, multinomial_nb, sgd_classifier, passive_aggressive"
        ),
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=100_000,
        help="Maximum word TF-IDF vocabulary size. Lower this to use less memory (default: 40000).",
    )
    parser.add_argument(
        "--char-features",
        type=int,
        default=65536,
        help="Number of hashed character features for URL robustness (default: 16384).",
    )
    parser.add_argument(
        "--word-min-df",
        type=int,
        default=5,
        help="Minimum document frequency for word n-grams (default: 5).",
    )
    parser.add_argument(
        "--char-max-chars",
        type=int,
        default=512,
        help="Maximum number of characters per sample for char features (default: 512).",
    )
    parser.add_argument(
        "--disable-char-features",
        action="store_true",
        help="Disable character hashing features to minimize memory usage.",
    )
    parser.add_argument(
        "--prefer-calibrated",
        action="store_true",
        help=(
            "Prefer calibrated_linear_svc for production if its selection metric is close to the top model. "
            "Useful when you want better confidence calibration."
        ),
    )
    parser.add_argument(
        "--calibrated-margin",
        type=float,
        default=0.002,
        help=(
            "Maximum score gap allowed when preferring calibrated_linear_svc over the top-ranked model "
            "(default: 0.002)."
        ),
    )
    parser.add_argument(
        "--disable-url-augmentation",
        action="store_true",
        help="Disable URL hard-example augmentation (typosquatting/spoofing/urgent/shortener patterns).",
    )
    parser.add_argument(
        "--url-augmentation-weight",
        type=int,
        default=2,
        help="How strongly to weight curated URL hard examples (default: 2).",
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


def prepare_training_data(
    email_path: Path,
    url_path: Path,
    use_url_augmentation: bool = True,
    url_augmentation_weight: int = 2,
) -> tuple[pd.DataFrame, dict[str, int]]:
    email_df = load_email_dataset(email_path)
    url_df = load_url_dataset(url_path)

    base_rows = int(len(email_df) + len(url_df))
    combined = pd.concat([email_df, url_df], axis=0, ignore_index=True)

    augmentation_rows_added = 0
    if use_url_augmentation:
        augmented_url_df = _build_augmented_url_dataset(repeat=url_augmentation_weight)
        augmentation_rows_added = int(len(augmented_url_df))
        combined = pd.concat([combined, augmented_url_df], axis=0, ignore_index=True)

    combined["text"] = combined["text"].fillna("").astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0].copy()
    combined["text"] = combined["text"].str.replace(r"\s+", "", regex=True).where(
        combined["source"].str.contains("url"),
        combined["text"],
    )
    combined["label"] = normalize_binary_labels(combined["label"])
    combined = combined.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    augmentation_summary = {
        "base_rows": base_rows,
        "augmented_rows_added": augmentation_rows_added,
        "total_rows_after_dedup": int(len(combined)),
    }

    return combined, augmentation_summary


def build_word_vectorizer(max_features: int = 100_000, min_df: int = 5) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.95,
        sublinear_tf=True,
        max_features=max_features,
    )


def build_char_vectorizer(n_features: int = 16_384, max_chars: int = 512) -> Pipeline:
    # Hashing avoids building a giant character vocabulary in memory.
    return Pipeline(
        steps=[
            (
                "char_truncate",
                FunctionTransformer(
                    partial(truncate_for_char_features, max_chars=max_chars),
                    validate=False,
                ),
            ),
            (
                "char_hash",
                HashingVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    n_features=n_features,
                    alternate_sign=False,
                    lowercase=True,
                    norm="l2",
                    binary=True,
                    dtype=np.float64,
                ),
            ),
        ]
    )


def build_vectorizer(
    max_features: int = 100_000,
    char_features: int = 65_536,
    word_min_df: int = 5,
    char_max_chars: int = 512,
    use_char_features: bool = True,
) -> FeatureUnion:
    transformers: list[tuple[str, Any]] = [
        ("word_tfidf", build_word_vectorizer(max_features=max_features, min_df=word_min_df))
    ]
    if use_char_features:
        transformers.append(
            (
                "char_tfidf",
                build_char_vectorizer(n_features=char_features, max_chars=char_max_chars),
            )
        )

    return FeatureUnion(transformer_list=transformers)


def get_model_candidates(
    random_state: int,
    max_features: int = 100_000,
    char_features: int = 65_536,
    word_min_df: int = 5,
    char_max_chars: int = 512,
    use_char_features: bool = True,
) -> dict[str, Callable[[], BaseEstimator]]:
    return {
        "logistic_regression": lambda: Pipeline(
            steps=[
                (
                    "tfidf",
                    build_vectorizer(
                        max_features=max_features,
                        char_features=char_features,
                        word_min_df=word_min_df,
                        char_max_chars=char_max_chars,
                        use_char_features=use_char_features,
                    ),
                ),
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
                (
                    "tfidf",
                    build_vectorizer(
                        max_features=max_features,
                        char_features=char_features,
                        word_min_df=word_min_df,
                        char_max_chars=char_max_chars,
                        use_char_features=use_char_features,
                    ),
                ),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "calibrated_linear_svc": lambda: Pipeline(
            steps=[
                (
                    "tfidf",
                    build_vectorizer(
                        max_features=max_features,
                        char_features=char_features,
                        word_min_df=word_min_df,
                        char_max_chars=char_max_chars,
                        use_char_features=use_char_features,
                    ),
                ),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
        "multinomial_nb": lambda: Pipeline(
            steps=[
                (
                    "tfidf",
                    build_vectorizer(
                        max_features=max_features,
                        char_features=char_features,
                        word_min_df=word_min_df,
                        char_max_chars=char_max_chars,
                        use_char_features=use_char_features,
                    ),
                ),
                ("clf", MultinomialNB(alpha=0.5)),
            ]
        ),
        "sgd_classifier": lambda: Pipeline(
            steps=[
                (
                    "tfidf",
                    build_vectorizer(
                        max_features=max_features,
                        char_features=char_features,
                        word_min_df=word_min_df,
                        char_max_chars=char_max_chars,
                        use_char_features=use_char_features,
                    ),
                ),
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
                (
                    "tfidf",
                    build_vectorizer(
                        max_features=max_features,
                        char_features=char_features,
                        word_min_df=word_min_df,
                        char_max_chars=char_max_chars,
                        use_char_features=use_char_features,
                    ),
                ),
                (
                    "clf",
                    SGDClassifier(
                        loss="hinge",
                        penalty=None,
                        learning_rate="pa1",
                        eta0=1.0,
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


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    source_true: pd.Series | None = None,
) -> dict[str, Any]:
    precision, recall, phishing_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    matrix = confusion_matrix(y_true, y_pred).tolist()
    full_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "precision_phishing": float(precision),
        "recall_phishing": float(recall),
        "phishing_f1": float(phishing_f1),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": matrix,
        "classification_report": full_report,
    }

    if source_true is not None:
        source_series = source_true.reset_index(drop=True)
        y_true_series = pd.Series(y_true).reset_index(drop=True)
        y_pred_series = pd.Series(y_pred).reset_index(drop=True)

        per_source: dict[str, Any] = {}
        for source_name in sorted(source_series.unique().tolist()):
            mask = source_series == source_name
            if mask.sum() == 0:
                continue

            ys = y_true_series[mask]
            yp = y_pred_series[mask]
            p, r, f1, _ = precision_recall_fscore_support(
                ys,
                yp,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
            per_source[source_name] = {
                "rows": int(mask.sum()),
                "accuracy": float(accuracy_score(ys, yp)),
                "precision_phishing": float(p),
                "recall_phishing": float(r),
                "phishing_f1": float(f1),
                "label_distribution": {
                    "legitimate_0": int((ys == 0).sum()),
                    "phishing_1": int((ys == 1).sum()),
                },
            }

        metrics["per_source"] = per_source

    return metrics


def train_and_evaluate_models(
    x_train: pd.Series,
    y_train: pd.Series,
    x_test: pd.Series,
    y_test: pd.Series,
    source_test: pd.Series,
    model_builders: dict[str, Callable[[], BaseEstimator]],
    requested_models: list[str],
) -> tuple[list[dict[str, Any]], dict[str, BaseEstimator]]:
    results: list[dict[str, Any]] = []
    fitted_models: dict[str, BaseEstimator] = {}

    for model_name in requested_models:
        model = model_builders[model_name]()

        train_start = perf_counter()
        try:
            model.fit(x_train, y_train)
        except MemoryError as exc:
            raise MemoryError(
                "Training ran out of memory. Try lower settings such as: "
                "--max-features 20000 --char-features 8192 --word-min-df 8 --char-max-chars 256 "
                "or add --disable-char-features."
            ) from exc
        train_seconds = perf_counter() - train_start

        predict_start = perf_counter()
        y_pred = model.predict(x_test)
        predict_seconds = perf_counter() - predict_start

        metrics = evaluate_predictions(y_test, y_pred, source_true=source_test)
        metrics["name"] = model_name
        metrics["train_seconds"] = float(train_seconds)
        metrics["predict_seconds"] = float(predict_seconds)

        results.append(metrics)
        fitted_models[model_name] = model
        gc.collect()

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


def select_production_model(
    ranked_results: list[dict[str, Any]],
    selection_metric: str,
    prefer_calibrated: bool,
    calibrated_margin: float,
) -> tuple[str, dict[str, Any]]:
    top = ranked_results[0]
    selected = top
    reason = "top_ranked"

    if prefer_calibrated:
        calibrated = next((item for item in ranked_results if item["name"] == "calibrated_linear_svc"), None)
        if calibrated is not None:
            gap = float(top[selection_metric]) - float(calibrated[selection_metric])
            if gap <= calibrated_margin:
                selected = calibrated
                reason = "calibrated_within_margin"

    return selected["name"], {
        "strategy": "prefer_calibrated_if_close" if prefer_calibrated else "top_ranked",
        "reason": reason,
        "calibrated_margin": float(calibrated_margin),
    }


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
    if args.calibrated_margin < 0:
        raise ValueError("--calibrated-margin must be >= 0")
    if args.url_augmentation_weight < 1:
        raise ValueError("--url-augmentation-weight must be >= 1")

    df, augmentation_summary = prepare_training_data(
        email_path=email_path,
        url_path=url_path,
        use_url_augmentation=not args.disable_url_augmentation,
        url_augmentation_weight=args.url_augmentation_weight,
    )

    x_train, x_test, y_train, y_test, _source_train, source_test = train_test_split(
        df["text"],
        df["label"],
        df["source"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    model_builders = get_model_candidates(
        random_state=args.random_state,
        max_features=args.max_features,
        char_features=args.char_features,
        word_min_df=args.word_min_df,
        char_max_chars=args.char_max_chars,
        use_char_features=not args.disable_char_features,
    )
    requested_models = parse_requested_models(args.models, model_builders)

    results, fitted_models = train_and_evaluate_models(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        source_test=source_test,
        model_builders=model_builders,
        requested_models=requested_models,
    )
    ranked_results = rank_models(results, selection_metric=args.selection_metric)

    best_model_name, selection_details = select_production_model(
        ranked_results=ranked_results,
        selection_metric=args.selection_metric,
        prefer_calibrated=args.prefer_calibrated,
        calibrated_margin=args.calibrated_margin,
    )
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
            "augmentation": {
                "enabled": not args.disable_url_augmentation,
                "weight": int(args.url_augmentation_weight),
                **augmentation_summary,
            },
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
            "details": selection_details,
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

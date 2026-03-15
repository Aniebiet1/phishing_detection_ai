"""
Quick offline model test script.
Usage:
    python test_model.py
    python test_model.py --model saved_models/phishing_model.joblib
    python test_model.py --text "Click here to claim your prize!"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test / inspect a saved phishing model.")
    parser.add_argument(
        "--model",
        type=str,
        default="saved_models/phishing_model.joblib",
        help="Path to the joblib model file.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="saved_models/training_report.json",
        help="Path to training_report.json (shows saved metrics).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text / URL to classify interactively.",
    )
    return parser.parse_args()


SAMPLE_TESTS = [
    # (text, expected_label)
    ("Congratulations! You have won a $1000 Walmart gift card. Click now.", 1),
    ("https://secure-login-paypal.malicious-site.com/verify", 1),
    ("http://www.paypal.com", 0),
    ("Meeting notes from the all-hands call on Thursday.", 0),
    ("URGENT: Your bank account has been suspended. Verify immediately.", 1),
    ("https://www.google.com/search?q=python+tutorial", 0),
    ("Your OTP for login is 483920. Valid for 5 minutes.", 0),
    ("WIN FREE IPHONE 15! Limited slots. Click the link below NOW!", 1),
]


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"[ERROR] Model not found at '{model_path}'.")
        print("        Run  python train.py  first to generate it.")
        return

    model = joblib.load(model_path)
    print(f"[OK] Model loaded from: {model_path}")

    # Show saved training report metrics if available
    report_path = Path(args.report)
    if report_path.exists():
        with report_path.open(encoding="utf-8") as f:
            report = json.load(f)

        print_section("Training Report Summary")
        print(f"  Timestamp : {report.get('timestamp_utc', 'n/a')}")

        data = report.get("data", {})
        if data:
            print(f"  Rows used : {data.get('rows_used', 'n/a')}")
            dist = data.get("label_distribution", {})
            print(f"  Labels    : legitimate={dist.get('legitimate_0', 'n/a')}  phishing={dist.get('phishing_1', 'n/a')}")

        selection = report.get("selection", {})
        if selection:
            print(f"  Best model: {selection.get('best_model', 'n/a')}  (metric: {selection.get('metric', 'n/a')})")

        models_list = report.get("models", [])
        if models_list:
            print_section("All Model Scores (ranked)")
            header = f"  {'Model':<24} {'Accuracy':>8} {'Phish_F1':>9} {'Macro_F1':>9}  {'Train(s)':>8}"
            print(header)
            print("  " + "-" * 62)
            for m in models_list:
                print(
                    f"  {m['name']:<24} {m['accuracy']:>8.4f} {m['phishing_f1']:>9.4f} "
                    f"{m['macro_f1']:>9.4f}  {m.get('train_seconds', 0.0):>8.2f}s"
                )

    # Interactive single-text prediction
    if args.text:
        print_section("Interactive Prediction")
        pred = model.predict([args.text])[0]
        label = "phishing" if int(pred) == 1 else "legitimate"
        score = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([args.text])[0]
            score = float(max(proba))
        print(f"  Input : {args.text[:80]}")
        print(f"  Label : {label}  (raw={pred})")
        if score is not None:
            print(f"  Score : {score:.4f}")
        return

    # Run built-in sample tests
    print_section("Built-in Sample Tests")
    texts = [t for t, _ in SAMPLE_TESTS]
    expected = [e for _, e in SAMPLE_TESTS]
    predictions = model.predict(texts)

    passed = 0
    for text, exp, pred in zip(texts, expected, predictions):
        status = "PASS" if int(pred) == exp else "FAIL"
        pred_label = "phishing" if int(pred) == 1 else "legitimate"
        exp_label  = "phishing" if exp == 1 else "legitimate"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] pred={pred_label:<12} expected={exp_label:<12} | {text[:55]}")

    print(f"\n  Result: {passed}/{len(SAMPLE_TESTS)} passed")

    print_section("Classification Report (sample tests)")
    print(classification_report(expected, predictions, target_names=["legitimate", "phishing"]))

    print_section("Confusion Matrix (sample tests)")
    cm = confusion_matrix(expected, predictions)
    print(f"                 Predicted")
    print(f"                 legit  phish")
    print(f"  Actual legit   {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"  Actual phish   {cm[1][0]:>5}  {cm[1][1]:>5}")


if __name__ == "__main__":
    main()

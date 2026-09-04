"""
Evaluate a trained multilingual sentiment model.

This script:
1. Loads a saved model checkpoint
2. Evaluates on each language-specific test set
3. Reports comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)
4. Supports loading real Tamil evaluation data
5. Compares across experiments

Usage:
    python scripts/evaluate_multilingual.py --model-dir models/indicbert_multilingual
    python scripts/evaluate_multilingual.py --model-dir models/indicbert_multilingual --real-tamil data/real_tamil.csv
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {"model": {"max_length": 128}, "data": {"output_dir": "data"}}


def predict_batch(texts, model, tokenizer, device, max_length=128, batch_size=32):
    """Run prediction on a list of texts."""
    all_predictions = []
    all_probabilities = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_predictions.extend(preds.cpu().numpy())
        all_probabilities.extend(probs.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities)


def evaluate_split(
    df, model, tokenizer, device, max_length, split_name="test"
):
    """Evaluate model on a dataframe split."""
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    label_names = ["negative", "neutral", "positive"]

    texts = df["text"].tolist()
    true_labels = df["sentiment"].map(label_map).values

    pred_labels, probabilities = predict_batch(
        texts, model, tokenizer, device, max_length
    )

    acc = accuracy_score(true_labels, pred_labels)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )

    # Per-class F1
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=None,
        labels=[0, 1, 2], zero_division=0,
    )

    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])

    report = classification_report(
        true_labels, pred_labels,
        target_names=label_names,
        zero_division=0,
    )

    return {
        "split": split_name,
        "samples": len(df),
        "accuracy": float(acc),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "macro_f1": float(f1_m),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "weighted_f1": float(f1_w),
        "per_class_f1": {
            label_names[i]: float(f1_per_class[i]) for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multilingual sentiment model"
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to saved model directory"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--real-tamil", type=str, default=None,
        help="Path to real Tamil evaluation dataset CSV"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory for results"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    max_length = config.get("model", {}).get("max_length", 128)
    data_dir = os.path.join(
        PROJECT_ROOT, config.get("data", {}).get("output_dir", "data")
    )

    # Resolve model directory
    model_dir = args.model_dir
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(PROJECT_ROOT, model_dir)

    # Output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(model_dir).replace("models", "results"),
            os.path.basename(model_dir),
        )
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=True
    ).to(device)
    model.eval()
    print("Model loaded successfully.")

    # Load test data
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_path):
        print(f"Error: Test dataset not found at {test_path}")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    print(f"Loaded test set: {len(test_df)} samples")

    # ===== Per-language evaluation =====
    all_results = {}

    for lang in sorted(test_df["language"].unique()):
        lang_df = test_df[test_df["language"] == lang]
        if len(lang_df) == 0:
            continue

        print(f"\nEvaluating {lang.upper()} ({len(lang_df)} samples)...")
        results = evaluate_split(
            lang_df, model, tokenizer, device, max_length, f"test_{lang}"
        )
        all_results[lang] = results

        print(f"  Accuracy:    {results['accuracy']:.4f}")
        print(f"  Macro F1:    {results['macro_f1']:.4f}")
        print(f"  Weighted F1: {results['weighted_f1']:.4f}")
        print(f"\n{results['classification_report']}")

    # ===== Combined evaluation =====
    print(f"\nEvaluating COMBINED ({len(test_df)} samples)...")
    combined_results = evaluate_split(
        test_df, model, tokenizer, device, max_length, "test_combined"
    )
    all_results["combined"] = combined_results

    print(f"  Accuracy:    {combined_results['accuracy']:.4f}")
    print(f"  Macro F1:    {combined_results['macro_f1']:.4f}")
    print(f"  Weighted F1: {combined_results['weighted_f1']:.4f}")
    print(f"\n{combined_results['classification_report']}")

    # ===== Real Tamil evaluation =====
    real_tamil_path = (
        args.real_tamil
        or config.get("evaluation", {}).get("real_tamil_dataset")
    )
    if real_tamil_path and os.path.exists(real_tamil_path):
        print(f"\nEvaluating REAL TAMIL ({real_tamil_path})...")
        real_tamil_df = pd.read_csv(real_tamil_path)

        # Validate schema
        if "text" in real_tamil_df.columns and "sentiment" in real_tamil_df.columns:
            real_results = evaluate_split(
                real_tamil_df, model, tokenizer, device, max_length,
                "test_real_tamil"
            )
            all_results["real_tamil"] = real_results

            print(f"  Accuracy:    {real_results['accuracy']:.4f}")
            print(f"  Macro F1:    {real_results['macro_f1']:.4f}")
            print(f"\n{real_results['classification_report']}")

            # Compare machine-translated vs real Tamil
            if "ta" in all_results:
                print("\n--- Machine-Translated vs Real Tamil ---")
                mt_f1 = all_results["ta"]["macro_f1"]
                real_f1 = real_results["macro_f1"]
                print(f"  Machine-translated Tamil F1: {mt_f1:.4f}")
                print(f"  Real Tamil F1:               {real_f1:.4f}")
                print(f"  Difference:                  {real_f1 - mt_f1:+.4f}")
        else:
            print(f"  Warning: Real Tamil dataset missing 'text' or 'sentiment' columns")

    # ===== Summary Table =====
    print(f"\n{'='*90}")
    print("EVALUATION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Language':<15} {'Samples':>8} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 90)

    for lang_key in ["en", "ta", "en-ta", "real_tamil", "combined"]:
        if lang_key in all_results:
            r = all_results[lang_key]
            name = {
                "en": "English",
                "ta": "Tamil (MT)",
                "en-ta": "Code-Mixed",
                "real_tamil": "Tamil (Real)",
                "combined": "COMBINED",
            }.get(lang_key, lang_key)
            print(f"{name:<15} {r['samples']:>8} {r['accuracy']:>10.4f} "
                  f"{r['precision_macro']:>10.4f} {r['recall_macro']:>10.4f} "
                  f"{r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f}")

    print(f"{'='*90}")

    # ===== Save Results =====
    # Remove classification_report strings for JSON serialization
    json_results = {}
    for k, v in all_results.items():
        json_results[k] = {
            key: val for key, val in v.items()
            if key != "classification_report"
        }

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=4)
    print(f"\nResults saved: {results_path}")

    # Save human-readable report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for lang_key, results in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"{lang_key.upper()} ({results['samples']} samples)\n")
            f.write(f"{'='*60}\n")
            f.write(results["classification_report"])
            f.write(f"\nConfusion Matrix:\n{np.array(results['confusion_matrix'])}\n")
    print(f"Report saved: {report_path}")

    print(f"\n[OK] Evaluation complete!")

    return all_results


if __name__ == "__main__":
    main()

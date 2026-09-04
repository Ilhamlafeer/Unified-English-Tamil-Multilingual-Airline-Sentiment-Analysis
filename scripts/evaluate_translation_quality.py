"""
Translation Quality Evaluation Utility.

This script:
1. Samples N translated tweets for manual evaluation
2. Generates a CSV template for human annotation
3. After annotation, calculates quality percentages

Usage:
    # Generate annotation template
    python scripts/evaluate_translation_quality.py --generate --samples 200

    # Calculate results after manual annotation
    python scripts/evaluate_translation_quality.py --calculate --input data/translation_quality_annotated.csv
"""

import os
import sys
import argparse

import pandas as pd
import numpy as np

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
        return {
            "evaluation": {"translation_quality_samples": 200},
            "training": {"random_seed": 42},
            "data": {"output_dir": "data"},
        }


def generate_annotation_template(config, n_samples, output_path):
    """Generate a CSV template for manual translation quality evaluation."""
    data_dir = os.path.join(
        PROJECT_ROOT, config.get("data", {}).get("output_dir", "data")
    )
    tamil_path = os.path.join(data_dir, "tamil_dataset.csv")

    if not os.path.exists(tamil_path):
        print(f"Error: Tamil dataset not found at {tamil_path}")
        sys.exit(1)

    tamil_df = pd.read_csv(tamil_path)
    print(f"Loaded {len(tamil_df)} Tamil translations")

    seed = config.get("training", {}).get("random_seed", 42)

    # Stratified sampling by sentiment
    n_per_sentiment = n_samples // 3
    remainder = n_samples - n_per_sentiment * 3

    samples = []
    for i, sentiment in enumerate(["negative", "neutral", "positive"]):
        sent_df = tamil_df[tamil_df["sentiment"] == sentiment]
        n = n_per_sentiment + (1 if i < remainder else 0)
        n = min(n, len(sent_df))
        samples.append(sent_df.sample(n=n, random_state=seed))

    sample_df = pd.concat(samples, ignore_index=True)

    # Create annotation template
    template = pd.DataFrame({
        "original_tweet_id": sample_df["original_tweet_id"],
        "original_text": sample_df["original_text"],
        "translated_text": sample_df["text"],
        "sentiment": sample_df["sentiment"],
        "translation_correct": "",      # Yes/No
        "meaning_preserved": "",         # Yes/No
        "sentiment_preserved": "",       # Yes/No
        "natural_tamil": "",             # Yes/No/Partial
        "notes": "",
    })

    template.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[OK] Annotation template saved: {output_path}")
    print(f"   Samples: {len(template)}")
    print(f"\n   Distribution:")
    for sent, count in template["sentiment"].value_counts().items():
        print(f"     {sent}: {count}")

    print(f"\nInstructions:")
    print(f"  1. Open {output_path} in a spreadsheet editor")
    print(f"  2. For each row, fill in:")
    print(f"     - translation_correct: Yes/No")
    print(f"     - meaning_preserved: Yes/No")
    print(f"     - sentiment_preserved: Yes/No")
    print(f"     - natural_tamil: Yes/No/Partial")
    print(f"     - notes: any additional observations")
    print(f"  3. Save the file")
    print(f"  4. Run: python scripts/evaluate_translation_quality.py "
          f"--calculate --input {output_path}")


def calculate_quality_metrics(input_path):
    """Calculate translation quality metrics from annotated CSV."""
    if not os.path.exists(input_path):
        print(f"Error: Annotated file not found at {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} annotated translations")

    # Calculate metrics
    metrics = {}
    for col in ["translation_correct", "meaning_preserved",
                 "sentiment_preserved", "natural_tamil"]:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found")
            continue

        values = df[col].dropna().astype(str).str.strip().str.lower()
        total = len(values)

        if total == 0:
            print(f"Warning: No annotations in column '{col}'")
            continue

        yes_count = values.isin(["yes", "y", "1", "true"]).sum()
        no_count = values.isin(["no", "n", "0", "false"]).sum()
        partial_count = values.isin(["partial", "p"]).sum()
        annotated = yes_count + no_count + partial_count
        unannotated = total - annotated

        metrics[col] = {
            "total": total,
            "annotated": annotated,
            "yes": yes_count,
            "no": no_count,
            "partial": partial_count,
            "unannotated": unannotated,
            "yes_percentage": (yes_count / annotated * 100) if annotated > 0 else 0,
        }

    # Print results
    print(f"\n{'='*60}")
    print("TRANSLATION QUALITY EVALUATION RESULTS")
    print(f"{'='*60}")

    for col, m in metrics.items():
        col_display = col.replace("_", " ").title()
        print(f"\n{col_display}:")
        print(f"  Annotated:  {m['annotated']} / {m['total']}")
        print(f"  Yes:        {m['yes']} ({m['yes_percentage']:.1f}%)")
        print(f"  No:         {m['no']}")
        if m["partial"] > 0:
            print(f"  Partial:    {m['partial']}")

    # Per-sentiment breakdown
    print(f"\n--- Per-Sentiment Breakdown ---")
    if "sentiment" in df.columns and "sentiment_preserved" in df.columns:
        for sentiment in ["negative", "neutral", "positive"]:
            sent_df = df[df["sentiment"] == sentiment]
            vals = sent_df["sentiment_preserved"].dropna().astype(str).str.lower()
            total = len(vals)
            yes = vals.isin(["yes", "y", "1", "true"]).sum()
            pct = (yes / total * 100) if total > 0 else 0
            print(f"  {sentiment}: {yes}/{total} preserved ({pct:.1f}%)")

    print(f"\n{'='*60}")

    # Save metrics
    output_path = input_path.replace(".csv", "_metrics.json")
    import json
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved: {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Translation quality evaluation utility"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate annotation template"
    )
    parser.add_argument(
        "--calculate", action="store_true",
        help="Calculate quality metrics from annotated CSV"
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Number of samples for annotation"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to annotated CSV (for --calculate)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for annotation template output"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.generate:
        n_samples = (
            args.samples
            or config.get("evaluation", {}).get("translation_quality_samples", 200)
        )
        output_path = (
            args.output
            or os.path.join(
                PROJECT_ROOT,
                config.get("data", {}).get("output_dir", "data"),
                "translation_quality_template.csv",
            )
        )
        generate_annotation_template(config, n_samples, output_path)

    elif args.calculate:
        if args.input is None:
            print("Error: --input required with --calculate")
            sys.exit(1)
        calculate_quality_metrics(args.input)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

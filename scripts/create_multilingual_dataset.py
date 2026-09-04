"""
Combine English + Tamil + Code-Mixed into a unified multilingual dataset.

This script:
1. Loads all three language datasets
2. Enforces data leakage prevention (same original_tweet_id → same split)
3. Applies language balancing if configured
4. Creates the final multilingual_dataset.csv and per-split CSVs
5. Reports detailed statistics

Usage:
    python scripts/create_multilingual_dataset.py
    python scripts/create_multilingual_dataset.py --config config.yaml
    python scripts/create_multilingual_dataset.py --no-balance
"""

import os
import sys
import json
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
            "data": {"output_dir": "data"},
            "language_balance": {"enabled": True, "target_ratios": None},
            "training": {"random_seed": 42},
        }


def verify_no_leakage(df):
    """
    Verify that all variants of the same original_tweet_id
    are in the same split.

    Returns True if no leakage detected, False otherwise.
    """
    print("\n--- Data Leakage Check ---")

    # Group by original_tweet_id and check that each group has exactly one split
    id_splits = df.groupby("original_tweet_id")["split"].nunique()
    leaky_ids = id_splits[id_splits > 1]

    if len(leaky_ids) == 0:
        print("  [OK] No data leakage detected!")
        print(f"     All {df['original_tweet_id'].nunique()} original tweets "
              f"have consistent splits across language variants")
        return True
    else:
        print(f"  [ERROR] DATA LEAKAGE DETECTED in {len(leaky_ids)} original tweets!")
        for tid in leaky_ids.head(5).index:
            subset = df[df["original_tweet_id"] == tid][["language", "split"]]
            print(f"     Tweet {tid}: {subset.to_dict('records')}")
        return False


def balance_languages(df, config):
    """
    Balance the dataset across languages.

    Strategy:
    - Determine the target count per language
    - Undersample majority languages or oversample minority languages
    - Maintain sentiment distribution within each language
    """
    balance_cfg = config.get("language_balance", {})
    seed = config.get("training", {}).get("random_seed", 42)

    if not balance_cfg.get("enabled", True):
        print("Language balancing disabled.")
        return df

    target_ratios = balance_cfg.get("target_ratios", None)

    print("\n--- Language Balancing ---")
    print("Before balancing:")
    for lang, count in df["language"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {lang:>5s}: {count:>6d} ({pct:.1f}%)")

    # Only balance training data; keep val/test as-is
    train_df = df[df["split"] == "train"].copy()
    other_df = df[df["split"] != "train"].copy()

    if target_ratios is None:
        # Default: match the smallest language count
        lang_counts = train_df["language"].value_counts()
        target_per_lang = lang_counts.min()
        print(f"\nTarget per language (train): {target_per_lang} (matching smallest)")
    else:
        # Use configured ratios
        total_train = len(train_df)
        target_per_lang = {}
        for lang, ratio in target_ratios.items():
            target_per_lang[lang] = int(total_train * ratio)
        print(f"\nTarget per language (train): {target_per_lang}")

    # Sample each language group
    balanced_parts = []
    for lang in train_df["language"].unique():
        lang_df = train_df[train_df["language"] == lang]

        if isinstance(target_per_lang, dict):
            target = target_per_lang.get(lang, len(lang_df))
        else:
            target = target_per_lang

        if len(lang_df) > target:
            # Undersample with stratification by sentiment
            sampled = lang_df.groupby("sentiment", group_keys=False).apply(
                lambda x: x.sample(
                    n=min(int(target * len(x) / len(lang_df)), len(x)),
                    random_state=seed,
                )
            )
            balanced_parts.append(sampled)
        elif len(lang_df) < target:
            # Oversample with stratification by sentiment
            n_extra = target - len(lang_df)
            extra = lang_df.sample(n=n_extra, replace=True, random_state=seed)
            balanced_parts.append(pd.concat([lang_df, extra], ignore_index=True))
        else:
            balanced_parts.append(lang_df)

    balanced_train = pd.concat(balanced_parts, ignore_index=True)

    # Recombine with val/test
    result = pd.concat([balanced_train, other_df], ignore_index=True)

    print("\nAfter balancing (train split only):")
    train_balanced = result[result["split"] == "train"]
    for lang, count in train_balanced["language"].value_counts().items():
        pct = count / len(train_balanced) * 100
        print(f"  {lang:>5s}: {count:>6d} ({pct:.1f}%)")

    return result


def print_statistics(df):
    """Print comprehensive dataset statistics."""
    print("\n" + "=" * 70)
    print("UNIFIED MULTILINGUAL DATASET STATISTICS")
    print("=" * 70)

    print(f"\nTotal records: {df.shape[0]}")
    print(f"Unique original tweets: {df['original_tweet_id'].nunique()}")

    # By language
    print("\n--- By Language ---")
    for lang, count in df["language"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {lang:>5s}: {count:>6d} ({pct:.1f}%)")

    # By sentiment
    print("\n--- By Sentiment ---")
    for sent, count in df["sentiment"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {sent:>10s}: {count:>6d} ({pct:.1f}%)")

    # By source
    print("\n--- By Source ---")
    for source, count in df["source"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {source:>25s}: {count:>6d} ({pct:.1f}%)")

    # By split
    print("\n--- By Split ---")
    for split, count in df["split"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {split:>12s}: {count:>6d} ({pct:.1f}%)")

    # Cross-tabulation: Language × Sentiment × Split
    print("\n--- Language × Sentiment × Split ---")
    cross = pd.crosstab(
        [df["split"], df["language"]],
        df["sentiment"],
        margins=True,
    )
    print(cross.to_string())

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Create unified multilingual dataset"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Disable language balancing"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    data_cfg = config.get("data", {})

    if args.no_balance:
        config["language_balance"] = {"enabled": False}

    # Paths
    output_dir = os.path.join(PROJECT_ROOT, data_cfg.get("output_dir", "data"))
    english_path = os.path.join(output_dir, "english_dataset.csv")
    tamil_path = os.path.join(output_dir, "tamil_dataset.csv")
    codemixed_path = os.path.join(output_dir, "codemixed_dataset.csv")

    multilingual_path = os.path.join(output_dir, "multilingual_dataset.csv")
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")

    # Load datasets
    datasets = []

    # English (required)
    if not os.path.exists(english_path):
        print(f"Error: English dataset not found at {english_path}")
        print("Run scripts/preprocess.py first.")
        sys.exit(1)
    english_df = pd.read_csv(english_path)
    print(f"Loaded English: {len(english_df)} records")
    datasets.append(english_df)

    # Tamil (optional but expected)
    if os.path.exists(tamil_path):
        tamil_df = pd.read_csv(tamil_path)
        # Drop translation_model column for unified schema if present
        if "translation_model" in tamil_df.columns:
            tamil_df = tamil_df.drop(columns=["translation_model"])
        print(f"Loaded Tamil: {len(tamil_df)} records")
        datasets.append(tamil_df)
    else:
        print(f"Warning: Tamil dataset not found at {tamil_path}")
        print("  Run scripts/translate_to_tamil.py to generate it.")

    # Code-mixed (optional but expected)
    if os.path.exists(codemixed_path):
        codemixed_df = pd.read_csv(codemixed_path)
        print(f"Loaded Code-Mixed: {len(codemixed_df)} records")
        datasets.append(codemixed_df)
    else:
        print(f"Warning: Code-mixed dataset not found at {codemixed_path}")
        print("  Run scripts/create_codemixed.py to generate it.")

    # Combine
    unified_columns = [
        "original_tweet_id", "original_text", "text",
        "sentiment", "language", "source", "split",
    ]

    combined_parts = []
    for ds in datasets:
        # Ensure all required columns exist
        for col in unified_columns:
            if col not in ds.columns:
                ds[col] = None
        combined_parts.append(ds[unified_columns])

    df = pd.concat(combined_parts, ignore_index=True)
    print(f"\nCombined dataset: {len(df)} records")

    # ===== Data Leakage Verification =====
    leakage_ok = verify_no_leakage(df)
    if not leakage_ok:
        print("\n[WARNING] DATA LEAKAGE DETECTED. Fixing by enforcing split from English dataset...")
        # Fix: use the English dataset's split assignments as ground truth
        id_to_split = (
            english_df.set_index("original_tweet_id")["split"].to_dict()
        )
        df["split"] = df["original_tweet_id"].map(id_to_split)
        df = df.dropna(subset=["split"])
        verify_no_leakage(df)

    # ===== Language Balancing =====
    df = balance_languages(df, config)

    # ===== Statistics =====
    print_statistics(df)

    # ===== Save Outputs =====
    os.makedirs(output_dir, exist_ok=True)

    # Full multilingual dataset
    df.to_csv(multilingual_path, index=False, encoding="utf-8")
    print(f"\nSaved: {multilingual_path} ({len(df)} rows)")

    # Per-split CSVs
    for split_name, split_path in [
        ("train", train_path),
        ("validation", val_path),
        ("test", test_path),
    ]:
        split_df = df[df["split"] == split_name]
        split_df.to_csv(split_path, index=False, encoding="utf-8")
        print(f"Saved: {split_path} ({len(split_df)} rows)")

    print("\n[OK] Unified multilingual dataset created!")

    return df


if __name__ == "__main__":
    main()

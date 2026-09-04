"""
Preprocess the Twitter US Airline Sentiment dataset.

This script:
1. Loads Tweets.csv and programmatically identifies text/sentiment columns
2. Performs conservative preprocessing (preserves sentiment-bearing info)
3. Runs exploratory analysis (class distribution, missing values, duplicates, length)
4. Performs stratified train/val/test split BEFORE any augmentation
5. Assigns original_tweet_id to every record
6. Outputs data/english_dataset.csv with unified schema

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --config config.yaml
"""

import os
import sys
import json
import re
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
        print(f"Warning: Config file not found at {config_path}, using defaults.")
        return {
            "data": {
                "raw_dataset": "Tweets.csv",
                "output_dir": "data",
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
            },
            "training": {"random_seed": 42},
        }


def identify_columns(df):
    """
    Programmatically identify the text and sentiment columns.

    Strategy:
    - Text column: longest average string length among object columns
    - Sentiment column: object column with values matching sentiment keywords
    """
    sentiment_keywords = {"negative", "neutral", "positive"}
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Find sentiment column
    sentiment_col = None
    for col in object_cols:
        unique_vals = set(df[col].dropna().str.lower().unique())
        if unique_vals.issubset(sentiment_keywords) and len(unique_vals) >= 2:
            sentiment_col = col
            break

    # If no exact match, look for column with 'sentiment' in name
    if sentiment_col is None:
        for col in object_cols:
            if "sentiment" in col.lower():
                sentiment_col = col
                break

    # Find text column: longest average string length
    text_col = None
    max_avg_len = 0
    for col in object_cols:
        if col == sentiment_col:
            continue
        avg_len = df[col].dropna().astype(str).str.len().mean()
        if avg_len > max_avg_len:
            max_avg_len = avg_len
            text_col = col

    print(f"Identified text column: '{text_col}' (avg length: {max_avg_len:.1f})")
    print(f"Identified sentiment column: '{sentiment_col}'")

    return text_col, sentiment_col


def conservative_preprocess(text):
    """
    Conservative text preprocessing that preserves sentiment-bearing information.

    Preserves:
    - Emojis, hashtags, mentions, airline names
    - Negation words (not, never, don't, etc.)
    - Punctuation (especially ! and ?)
    - Repeated characters (e.g., 'soooo bad')
    - Sentiment-bearing expressions and slang

    Only does:
    - Strip leading/trailing whitespace
    - Collapse multiple spaces to single space
    - Remove null/NaN-like strings
    """
    if not isinstance(text, str):
        return ""

    # Strip whitespace
    text = text.strip()

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text


def exploratory_analysis(df, text_col, sentiment_col):
    """Print exploratory data analysis."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Dataset shape
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Class distribution
    print("\n--- Class Distribution ---")
    class_dist = df[sentiment_col].value_counts()
    for label, count in class_dist.items():
        pct = count / len(df) * 100
        print(f"  {label:>10s}: {count:>6d} ({pct:>5.1f}%)")

    # Missing values
    print(f"\n--- Missing Values ---")
    text_missing = df[text_col].isna().sum()
    sent_missing = df[sentiment_col].isna().sum()
    print(f"  Text column:      {text_missing} missing")
    print(f"  Sentiment column: {sent_missing} missing")

    # Duplicates
    duplicates = df[text_col].duplicated().sum()
    print(f"\n--- Duplicates ---")
    print(f"  Duplicate tweets: {duplicates}")

    # Tweet length statistics
    lengths = df[text_col].dropna().astype(str).str.len()
    print(f"\n--- Tweet Length (characters) ---")
    print(f"  Min:    {lengths.min():>6.0f}")
    print(f"  Max:    {lengths.max():>6.0f}")
    print(f"  Mean:   {lengths.mean():>6.1f}")
    print(f"  Median: {lengths.median():>6.0f}")
    print(f"  Std:    {lengths.std():>6.1f}")

    # Word count statistics
    word_counts = df[text_col].dropna().astype(str).str.split().str.len()
    print(f"\n--- Tweet Length (words) ---")
    print(f"  Min:    {word_counts.min():>6.0f}")
    print(f"  Max:    {word_counts.max():>6.0f}")
    print(f"  Mean:   {word_counts.mean():>6.1f}")
    print(f"  Median: {word_counts.median():>6.0f}")

    print("\n" + "=" * 60)


def create_stratified_split(df, config):
    """
    Create stratified train/val/test split.

    CRITICAL: This split happens BEFORE any translation or augmentation
    to prevent data leakage.
    """
    seed = config.get("training", {}).get("random_seed", 42)
    data_cfg = config.get("data", {})
    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio = data_cfg.get("val_ratio", 0.15)
    test_ratio = data_cfg.get("test_ratio", 0.15)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    print(f"\n--- Stratified Split ---")
    print(f"  Train: {train_ratio:.0%}")
    print(f"  Val:   {val_ratio:.0%}")
    print(f"  Test:  {test_ratio:.0%}")
    print(f"  Seed:  {seed}")

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=df["sentiment"],
    )

    # Second split: val vs test
    test_fraction_of_remainder = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_fraction_of_remainder,
        random_state=seed,
        stratify=val_test_df["sentiment"],
    )

    # Assign split labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    print(f"\n  Train samples:      {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples:       {len(test_df)}")

    # Print distribution per split
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split_df["sentiment"].value_counts()
        print(f"\n  {split_name} distribution:")
        for label, count in dist.items():
            print(f"    {label:>10s}: {count:>5d}")

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Tweets.csv for multilingual sentiment analysis"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Paths
    data_cfg = config.get("data", {})
    raw_dataset = data_cfg.get("raw_dataset", "Tweets.csv")
    output_dir = os.path.join(PROJECT_ROOT, data_cfg.get("output_dir", "data"))

    raw_path = os.path.join(PROJECT_ROOT, raw_dataset)
    if not os.path.exists(raw_path):
        print(f"Error: Dataset not found at {raw_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # ===== 1. Load dataset =====
    print("Loading dataset...")
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    # ===== 2. Identify columns programmatically =====
    text_col, sentiment_col = identify_columns(df)

    if text_col is None or sentiment_col is None:
        print("Error: Could not identify text or sentiment columns.")
        sys.exit(1)

    # ===== 3. Exploratory analysis =====
    exploratory_analysis(df, text_col, sentiment_col)

    # ===== 4. Conservative preprocessing =====
    print("\nApplying conservative preprocessing...")

    # Keep only rows with valid text and sentiment
    df = df[[text_col, sentiment_col]].copy()
    df.columns = ["original_text", "sentiment"]

    # Drop rows with missing text or sentiment
    before = len(df)
    df = df.dropna(subset=["original_text", "sentiment"])
    dropped_na = before - len(df)
    print(f"  Dropped {dropped_na} rows with missing values")

    # Remove duplicate tweets
    before = len(df)
    df = df.drop_duplicates(subset=["original_text"])
    dropped_dup = before - len(df)
    print(f"  Dropped {dropped_dup} duplicate tweets")

    # Normalize sentiment labels to lowercase
    df["sentiment"] = df["sentiment"].str.lower().str.strip()

    # Verify only valid sentiment labels remain
    valid_sentiments = {"negative", "neutral", "positive"}
    invalid = set(df["sentiment"].unique()) - valid_sentiments
    if invalid:
        print(f"  Warning: removing {len(df[df['sentiment'].isin(invalid)])} rows "
              f"with invalid sentiments: {invalid}")
        df = df[df["sentiment"].isin(valid_sentiments)]

    # Apply conservative text preprocessing
    df["text"] = df["original_text"].apply(conservative_preprocess)

    # Remove empty texts after preprocessing
    before = len(df)
    df = df[df["text"].str.len() > 0]
    dropped_empty = before - len(df)
    if dropped_empty > 0:
        print(f"  Dropped {dropped_empty} empty tweets after preprocessing")

    print(f"\n  Final dataset size: {len(df)} tweets")

    # ===== 5. Assign original_tweet_id =====
    df = df.reset_index(drop=True)
    df["original_tweet_id"] = range(len(df))

    # ===== 6. Stratified split =====
    df = create_stratified_split(df, config)

    # ===== 7. Build unified schema =====
    english_df = pd.DataFrame({
        "original_tweet_id": df["original_tweet_id"],
        "original_text": df["original_text"],
        "text": df["text"],
        "sentiment": df["sentiment"],
        "language": "en",
        "source": "original",
        "split": df["split"],
    })

    # ===== 8. Save outputs =====
    # Save English dataset
    english_path = os.path.join(output_dir, "english_dataset.csv")
    english_df.to_csv(english_path, index=False, encoding="utf-8")
    print(f"\nSaved English dataset: {english_path} ({len(english_df)} rows)")

    # Save split IDs for data leakage prevention
    for split_name in ["train", "validation", "test"]:
        split_ids = english_df[english_df["split"] == split_name][
            "original_tweet_id"
        ].tolist()
        ids_path = os.path.join(output_dir, f"{split_name}_ids.json")
        with open(ids_path, "w") as f:
            json.dump(split_ids, f)
        print(f"Saved {split_name} IDs: {ids_path} ({len(split_ids)} IDs)")

    print("\n[OK] Preprocessing complete!")
    print(f"   Output directory: {output_dir}")

    return english_df


if __name__ == "__main__":
    main()

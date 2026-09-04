"""
Generate English-Tamil code-mixed dataset.

This script creates synthetic code-mixed tweets by combining English and Tamil
versions of the same tweet. The code-mixing strategy:
1. Takes English + Tamil pairs (from translation)
2. Randomly mixes words from both languages
3. Preserves domain-specific English terms (airline, flight, etc.)

IMPORTANT: These are synthetic code-mixed examples, NOT naturally occurring
Tamil tweets. They are explicitly marked as source='synthetic_codemixed'.

Usage:
    python scripts/create_codemixed.py
    python scripts/create_codemixed.py --config config.yaml
    python scripts/create_codemixed.py --dry-run
"""

import os
import sys
import re
import argparse
import random

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
            "codemixed": {
                "fraction": 0.5,
                "min_words": 5,
                "swap_probability": 0.3,
                "keep_english": [
                    "airline", "flight", "airport", "boarding",
                    "luggage", "baggage", "ticket", "seat",
                    "delay", "cancel", "service", "crew", "pilot",
                ],
            },
            "training": {"random_seed": 42},
            "data": {"output_dir": "data"},
        }


def is_tamil_char(char):
    """Check if a character is Tamil script."""
    return "\u0B80" <= char <= "\u0BFF"


def contains_tamil(text):
    """Check if text contains Tamil characters."""
    return any(is_tamil_char(c) for c in text)


def tokenize_simple(text):
    """Simple word tokenization preserving punctuation."""
    # Split on whitespace but keep punctuation attached
    tokens = text.split()
    return tokens


def create_codemixed_variant(
    english_text, tamil_text, config, rng
):
    """
    Create a code-mixed variant from English and Tamil texts.

    Strategy:
    - Align words roughly by position
    - For each position, randomly choose English or Tamil word
    - Keep domain-specific English terms in English
    - Keep sentiment-bearing English words in English
    """
    cm_cfg = config.get("codemixed", {})
    swap_prob = cm_cfg.get("swap_probability", 0.3)
    keep_english = set(
        w.lower() for w in cm_cfg.get("keep_english", [])
    )

    en_tokens = tokenize_simple(english_text)
    ta_tokens = tokenize_simple(tamil_text)

    # Skip if too short
    min_words = cm_cfg.get("min_words", 5)
    if len(en_tokens) < min_words:
        return None

    # Build code-mixed text
    # Strategy: Use Tamil as the base structure, but randomly keep
    # some English words (especially domain terms)
    result_tokens = []

    # Use the shorter length for alignment
    min_len = min(len(en_tokens), len(ta_tokens))

    for i in range(min_len):
        en_word = en_tokens[i]
        ta_word = ta_tokens[i]

        # Always keep domain-specific English terms
        if en_word.lower().strip(".,!?;:") in keep_english:
            result_tokens.append(en_word)
        # Always keep mentions and hashtags in English
        elif en_word.startswith("@") or en_word.startswith("#"):
            result_tokens.append(en_word)
        # Randomly choose between English and Tamil
        elif rng.random() < swap_prob:
            result_tokens.append(en_word)
        else:
            result_tokens.append(ta_word)

    # Append remaining tokens from whichever is longer
    if len(ta_tokens) > min_len:
        result_tokens.extend(ta_tokens[min_len:])
    elif len(en_tokens) > min_len:
        result_tokens.extend(en_tokens[min_len:])

    mixed_text = " ".join(result_tokens)

    # Verify it actually contains both languages
    has_tamil = contains_tamil(mixed_text)
    has_english = bool(re.search(r"[a-zA-Z]{2,}", mixed_text))

    if not (has_tamil and has_english):
        return None

    return mixed_text


def load_real_codemixed(path):
    """
    Load a real code-mixed dataset if available.

    Expected CSV schema:
        text, sentiment, language
    where language = 'en-ta'
    """
    if path is None or not os.path.exists(path):
        return pd.DataFrame()

    print(f"Loading real code-mixed dataset from {path}...")
    df = pd.read_csv(path)

    # Validate schema
    required = {"text", "sentiment"}
    if not required.issubset(set(df.columns)):
        print(f"Warning: Real code-mixed dataset missing columns: {required - set(df.columns)}")
        return pd.DataFrame()

    df["language"] = "en-ta"
    df["source"] = "real_codemixed"

    print(f"Loaded {len(df)} real code-mixed tweets")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate English-Tamil code-mixed dataset"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate and display samples without saving"
    )
    parser.add_argument(
        "--real-codemixed", type=str, default=None,
        help="Path to real code-mixed dataset CSV"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    cm_cfg = config.get("codemixed", {})
    data_cfg = config.get("data", {})
    seed = config.get("training", {}).get("random_seed", 42)

    rng = random.Random(seed)
    np.random.seed(seed)

    # Paths
    output_dir = os.path.join(PROJECT_ROOT, data_cfg.get("output_dir", "data"))
    english_path = os.path.join(output_dir, "english_dataset.csv")
    tamil_path = os.path.join(output_dir, "tamil_dataset.csv")
    codemixed_path = os.path.join(output_dir, "codemixed_dataset.csv")

    # Check prerequisites
    if not os.path.exists(english_path):
        print(f"Error: English dataset not found at {english_path}")
        print("Run scripts/preprocess.py first.")
        sys.exit(1)

    if not os.path.exists(tamil_path):
        print(f"Error: Tamil dataset not found at {tamil_path}")
        print("Run scripts/translate_to_tamil.py first.")
        sys.exit(1)

    # Load datasets
    print("Loading English dataset...")
    english_df = pd.read_csv(english_path)
    print(f"Loaded {len(english_df)} English tweets")

    print("Loading Tamil dataset...")
    tamil_df = pd.read_csv(tamil_path)
    print(f"Loaded {len(tamil_df)} Tamil tweets")

    # Merge on original_tweet_id to get EN-TA pairs
    pairs = english_df.merge(
        tamil_df[["original_tweet_id", "text"]],
        on="original_tweet_id",
        suffixes=("_en", "_ta"),
    )
    print(f"Found {len(pairs)} EN-TA pairs")

    # Select fraction for code-mixing
    fraction = cm_cfg.get("fraction", 0.5)
    n_to_mix = int(len(pairs) * fraction)

    # Sample tweets for code-mixing (stratified by sentiment)
    selected = pairs.groupby("sentiment", group_keys=False).apply(
        lambda x: x.sample(
            n=min(int(len(x) * fraction), len(x)),
            random_state=seed,
        )
    )
    print(f"Selected {len(selected)} tweets for code-mixing ({fraction:.0%} of pairs)")

    # Generate code-mixed variants
    codemixed_rows = []
    skipped = 0

    for _, row in selected.iterrows():
        mixed_text = create_codemixed_variant(
            row["text_en"], row["text_ta"], config, rng
        )

        if mixed_text is None:
            skipped += 1
            continue

        codemixed_rows.append({
            "original_tweet_id": row["original_tweet_id"],
            "original_text": row["original_text"],
            "text": mixed_text,
            "sentiment": row["sentiment"],
            "language": "en-ta",
            "source": "synthetic_codemixed",
            "split": row["split"],
        })

    print(f"Generated {len(codemixed_rows)} code-mixed tweets "
          f"(skipped {skipped} too short or single-language)")

    # Create DataFrame
    codemixed_df = pd.DataFrame(codemixed_rows)

    # Add real code-mixed data if available
    real_cm_df = load_real_codemixed(args.real_codemixed)
    if len(real_cm_df) > 0:
        codemixed_df = pd.concat(
            [codemixed_df, real_cm_df], ignore_index=True
        )
        print(f"Total code-mixed tweets (synthetic + real): {len(codemixed_df)}")

    # Print distribution
    if len(codemixed_df) > 0:
        print(f"\n--- Code-Mixed Dataset Statistics ---")
        print(f"Total: {len(codemixed_df)}")
        print(f"\nBy sentiment:")
        for sentiment, count in codemixed_df["sentiment"].value_counts().items():
            print(f"  {sentiment:>10s}: {count}")
        print(f"\nBy source:")
        for source, count in codemixed_df["source"].value_counts().items():
            print(f"  {source:>20s}: {count}")
        print(f"\nBy split:")
        for split, count in codemixed_df["split"].value_counts().items():
            print(f"  {split:>12s}: {count}")

    # Show samples
    if len(codemixed_df) > 0:
        print(f"\n--- Sample Code-Mixed Tweets ---")
        for _, row in codemixed_df.head(5).iterrows():
            print(f"\n  Original: {row['original_text'][:70]}...")
            print(f"  Mixed:    {row['text'][:70]}...")
            print(f"  Sentiment: {row['sentiment']}")

    if args.dry_run:
        print("\n[DRY RUN] Not saving to disk.")
        return codemixed_df

    # Save
    os.makedirs(output_dir, exist_ok=True)
    codemixed_df.to_csv(codemixed_path, index=False, encoding="utf-8")
    print(f"\n[OK] Code-mixed dataset saved: {codemixed_path} ({len(codemixed_df)} rows)")

    return codemixed_df


if __name__ == "__main__":
    main()

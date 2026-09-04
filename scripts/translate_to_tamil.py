"""
Translate English tweets to Tamil using AI4Bharat IndicTrans2.

This script:
1. Loads the English dataset (with split assignments)
2. Preserves @mentions, #hashtags, URLs, emojis before translation
3. Translates natural language content using IndicTrans2
4. Supports batch processing, GPU/CPU auto-detection
5. Saves intermediate results for resume capability
6. Caches results to avoid re-translating

Usage:
    python scripts/translate_to_tamil.py
    python scripts/translate_to_tamil.py --batch-size 16 --config config.yaml
"""

import os
import sys
import re
import argparse
import time

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

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
            "translation": {
                "model": "ai4bharat/indictrans2-en-indic-dist-200M",
                "source_lang": "eng_Latn",
                "target_lang": "tam_Taml",
                "batch_size": 32,
                "max_length": 256,
                "num_beams": 5,
                "save_every_n_batches": 50,
            },
            "data": {"output_dir": "data"},
        }


# Regex patterns for entities to preserve
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def extract_entities(text):
    """Extract entities that should be preserved during translation."""
    entities = {
        "urls": URL_PATTERN.findall(text),
        "mentions": MENTION_PATTERN.findall(text),
        "hashtags": HASHTAG_PATTERN.findall(text),
    }

    # Create a clean version for translation
    clean_text = text
    for url in entities["urls"]:
        clean_text = clean_text.replace(url, " ")
    # Keep mentions and hashtags as they may provide context
    # but mark them for re-insertion

    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    return clean_text, entities


def restore_entities(translated_text, entities):
    """Restore preserved entities into translated text."""
    result = translated_text

    # Append mentions at the beginning if they existed
    if entities["mentions"]:
        mentions_str = " ".join(entities["mentions"])
        result = mentions_str + " " + result

    # Append hashtags at the end if they existed
    if entities["hashtags"]:
        hashtags_str = " ".join(entities["hashtags"])
        result = result + " " + hashtags_str

    # Append URLs at the end
    if entities["urls"]:
        urls_str = " ".join(entities["urls"])
        result = result + " " + urls_str

    return result.strip()


def load_translation_model(config):
    """Load IndicTrans2 model and processor."""
    trans_cfg = config.get("translation", {})
    model_name = trans_cfg.get("model", "ai4bharat/indictrans2-en-indic-dist-200M")

    print(f"Loading translation model: {model_name}")

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        print("Using CPU (translation will be slower)")
        dtype = torch.float32

    # Load IndicTrans2
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    model.eval()

    # Load IndicProcessor for pre/post processing
    try:
        from IndicTransToolkit.processor import IndicProcessor
        ip = IndicProcessor(inference=True)
        print("IndicTransToolkit processor loaded successfully")
    except ImportError:
        print(
            "ERROR: IndicTransToolkit not installed.\n"
            "Install via: pip install git+https://github.com/VarunGumma/IndicTransToolkit"
        )
        sys.exit(1)

    return model, tokenizer, ip, device


def translate_batch(
    texts, model, tokenizer, ip, device, src_lang, tgt_lang, config
):
    """Translate a batch of texts from source to target language."""
    trans_cfg = config.get("translation", {})
    max_length = trans_cfg.get("max_length", 256)
    num_beams = trans_cfg.get("num_beams", 5)

    # IndicProcessor preprocessing
    batch = ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)

    # Tokenize
    inputs = tokenizer(
        batch,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    # Generate translations
    with torch.inference_mode():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=1,
        )

    # Decode
    try:
        with tokenizer.as_target_tokenizer():
            translations = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
    except AttributeError:
        # Fallback for newer transformers versions where as_target_tokenizer is removed
        translations = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # IndicProcessor postprocessing
    translations = ip.postprocess_batch(translations, lang=tgt_lang)

    return translations


def main():
    parser = argparse.ArgumentParser(
        description="Translate English tweets to Tamil using IndicTrans2"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of tweets to translate (for testing)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    trans_cfg = config.get("translation", {})
    data_cfg = config.get("data", {})

    batch_size = args.batch_size or trans_cfg.get("batch_size", 32)
    src_lang = trans_cfg.get("source_lang", "eng_Latn")
    tgt_lang = trans_cfg.get("target_lang", "tam_Taml")
    save_every = trans_cfg.get("save_every_n_batches", 50)

    # Paths
    output_dir = os.path.join(PROJECT_ROOT, data_cfg.get("output_dir", "data"))
    english_path = os.path.join(output_dir, "english_dataset.csv")
    tamil_path = os.path.join(output_dir, "tamil_dataset.csv")
    cache_path = os.path.join(output_dir, "tamil_translation_cache.csv")

    # Check English dataset exists
    if not os.path.exists(english_path):
        print(f"Error: English dataset not found at {english_path}")
        print("Run scripts/preprocess.py first.")
        sys.exit(1)

    # Load English dataset
    print("Loading English dataset...")
    english_df = pd.read_csv(english_path)
    print(f"Loaded {len(english_df)} English tweets")

    if args.limit:
        english_df = english_df.head(args.limit)
        print(f"Limited to {len(english_df)} tweets for testing")

    # Check for cached translations (resume capability)
    translated_ids = set()
    cached_rows = []
    if os.path.exists(cache_path):
        print(f"Loading cached translations from {cache_path}...")
        cache_df = pd.read_csv(cache_path)
        translated_ids = set(cache_df["original_tweet_id"].tolist())
        cached_rows = cache_df.to_dict("records")
        print(f"Found {len(translated_ids)} cached translations")

    # Filter out already translated tweets
    to_translate = english_df[
        ~english_df["original_tweet_id"].isin(translated_ids)
    ].copy()
    print(f"Tweets remaining to translate: {len(to_translate)}")

    if len(to_translate) == 0:
        print("All tweets already translated! Building final dataset...")
    else:
        # Load translation model
        model, tokenizer, ip, device = load_translation_model(config)

        # Translate in batches
        total_batches = (len(to_translate) + batch_size - 1) // batch_size
        print(f"\nTranslating {len(to_translate)} tweets in {total_batches} batches "
              f"(batch_size={batch_size})...")

        start_time = time.time()
        new_rows = []

        for batch_idx in tqdm(range(total_batches), desc="Translating"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(to_translate))
            batch_df = to_translate.iloc[batch_start:batch_end]

            # Extract entities and prepare clean texts
            clean_texts = []
            entity_list = []
            for _, row in batch_df.iterrows():
                clean_text, entities = extract_entities(row["text"])
                clean_texts.append(clean_text)
                entity_list.append(entities)

            try:
                # Translate the batch
                translations = translate_batch(
                    clean_texts, model, tokenizer, ip, device,
                    src_lang, tgt_lang, config,
                )

                # Restore entities and build records
                for i, (_, row) in enumerate(batch_df.iterrows()):
                    translated_text = restore_entities(
                        translations[i], entity_list[i]
                    )

                    new_rows.append({
                        "original_tweet_id": row["original_tweet_id"],
                        "original_text": row["original_text"],
                        "text": translated_text,
                        "sentiment": row["sentiment"],
                        "language": "ta",
                        "source": "machine_translation",
                        "translation_model": "IndicTrans2",
                        "split": row["split"],
                    })

            except Exception as e:
                import traceback
                print(f"\nError translating batch {batch_idx}: {e}")
                traceback.print_exc()
                print("Saving progress and continuing...")
                # Stop if we hit too many consecutive errors (prevent 24 min silent fails)
                if batch_idx > 5 and len(new_rows) == 0:
                    print("\n[!] Too many consecutive failures. Aborting translation early.")
                    break
                continue

            # Save intermediate results periodically
            if (batch_idx + 1) % save_every == 0:
                all_cached = cached_rows + new_rows
                cache_df = pd.DataFrame(all_cached, columns=[
                    "original_tweet_id", "original_text", "text",
                    "sentiment", "language", "source", "translation_model", "split",
                ])
                cache_df.to_csv(cache_path, index=False, encoding="utf-8")
                elapsed = time.time() - start_time
                tweets_done = len(cached_rows) + len(new_rows)
                rate = tweets_done / elapsed if elapsed > 0 else 0
                print(
                    f"\n  Saved checkpoint: {tweets_done} translations "
                    f"({rate:.1f} tweets/sec)"
                )

        # Save final cache
        all_rows = cached_rows + new_rows
        cache_df = pd.DataFrame(all_rows, columns=[
            "original_tweet_id", "original_text", "text",
            "sentiment", "language", "source", "translation_model", "split",
        ])
        cache_df.to_csv(cache_path, index=False, encoding="utf-8")

        elapsed = time.time() - start_time
        print(f"\nTranslation completed in {elapsed:.1f} seconds")
        print(f"Total translations: {len(all_rows)}")

    # Build final Tamil dataset
    tamil_df = pd.DataFrame(columns=[
        "original_tweet_id", "original_text", "text",
        "sentiment", "language", "source", "translation_model", "split",
    ])
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        try:
            tamil_df = pd.read_csv(cache_path)
        except pd.errors.EmptyDataError:
            pass

    # Save Tamil dataset
    tamil_df.to_csv(tamil_path, index=False, encoding="utf-8")
    print(f"\n[OK] Tamil dataset saved: {tamil_path} ({len(tamil_df)} rows)")

    # Print sample translations
    if len(tamil_df) > 0:
        print("\n--- Sample Translations ---")
        sample = tamil_df.head(5)
        for _, row in sample.iterrows():
            print(f"\n  EN: {row['original_text'][:80]}...")
            print(f"  TA: {row['text'][:80]}...")
            print(f"  Sentiment: {row['sentiment']}, Split: {row['split']}")

    return tamil_df


if __name__ == "__main__":
    main()

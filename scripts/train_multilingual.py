"""
Train a multilingual sentiment classifier using IndicBERT-v3-270M.

This script:
1. Loads the unified multilingual dataset
2. Fine-tunes a multilingual transformer for 3-class sentiment classification
3. Supports ablation experiments (English-only, EN+TA, EN+TA+CodeMixed)
4. Uses HuggingFace Trainer with mixed precision, early stopping
5. Saves model, tokenizer, and training logs

Usage:
    # Full multilingual training (Experiment C)
    python scripts/train_multilingual.py

    # Ablation experiments
    python scripts/train_multilingual.py --experiment A
    python scripts/train_multilingual.py --experiment B
    python scripts/train_multilingual.py --experiment C

    # Use alternative model
    python scripts/train_multilingual.py --model FacebookAI/xlm-roberta-base

    # Override via environment variable
    MODEL_NAME=FacebookAI/xlm-roberta-base python scripts/train_multilingual.py
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

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
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
        print(f"Warning: Config file not found at {config_path}")
        sys.exit(1)


def get_experiment_config(config, experiment_name):
    """Get experiment-specific configuration."""
    experiments = config.get("experiments", {})
    if experiment_name not in experiments:
        print(f"Unknown experiment: {experiment_name}")
        print(f"Available: {list(experiments.keys())}")
        sys.exit(1)

    exp = experiments[experiment_name]
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {experiment_name}: {exp['name']}")
    print(f"Description: {exp['description']}")
    print(f"Train languages: {exp['train_languages']}")
    print(f"{'='*60}")
    return exp


def load_dataset_for_experiment(data_dir, experiment_config):
    """Load and filter dataset based on experiment configuration."""
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            print("Run scripts/create_multilingual_dataset.py first.")
            sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Filter training data by experiment languages
    train_languages = experiment_config["train_languages"]
    train_df = train_df[train_df["language"].isin(train_languages)].copy()

    # Validation: use same languages as training
    val_df = val_df[val_df["language"].isin(train_languages)].copy()

    # Test: always use ALL languages for evaluation
    # (test_df is kept as-is)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)} (languages: {train_languages})")
    print(f"  Val:   {len(val_df)} (languages: {train_languages})")
    print(f"  Test:  {len(test_df)} (all languages)")

    # Print distribution
    print(f"\nTrain language distribution:")
    for lang, count in train_df["language"].value_counts().items():
        print(f"  {lang}: {count}")

    return train_df, val_df, test_df


def encode_labels(df):
    """Encode sentiment labels to integers."""
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df = df.copy()
    df["label"] = df["sentiment"].map(label_map)

    # Verify no unmapped labels
    unmapped = df["label"].isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} rows with unmapped sentiment labels")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df


def compute_metrics(eval_pred):
    """Compute evaluation metrics for the Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)

    # Macro metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    # Weighted metrics
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "macro_f1": f1_macro,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "weighted_f1": f1_w,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train multilingual sentiment classifier"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--experiment", type=str, default="C",
        choices=["A", "B", "C"],
        help="Ablation experiment: A=EN, B=EN+TA, C=EN+TA+CM"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name (e.g., FacebookAI/xlm-roberta-base)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})
    data_cfg = config.get("data", {})

    # Model name: CLI arg > env var > config
    model_name = (
        args.model
        or os.environ.get("MODEL_NAME")
        or model_cfg.get("name", "ai4bharat/IndicBERT-v3-270M")
    )

    # Training hyperparameters
    max_length = model_cfg.get("max_length", 128)
    learning_rate = train_cfg.get("learning_rate", 2e-5)
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    eval_batch_size = train_cfg.get("eval_batch_size", 16)
    epochs = args.epochs or train_cfg.get("epochs", 3)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    seed = train_cfg.get("random_seed", 42)
    use_fp16 = train_cfg.get("use_fp16", True)
    use_bf16 = train_cfg.get("use_bf16", False)
    patience = train_cfg.get("early_stopping_patience", 2)
    logging_steps = train_cfg.get("logging_steps", 50)
    save_total_limit = train_cfg.get("save_total_limit", 2)

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Disable fp16 on CPU
    else:
        use_fp16 = False
        use_bf16 = False
        print("Running on CPU - mixed precision disabled")

    # Get experiment configuration
    exp_config = get_experiment_config(config, args.experiment)

    # Paths
    data_dir = os.path.join(PROJECT_ROOT, data_cfg.get("output_dir", "data"))

    # Create model-specific output directory
    model_short = model_name.split("/")[-1].lower()
    lang_suffix = "_".join(exp_config["train_languages"]).replace("-", "")
    experiment_name = f"{model_short}_{lang_suffix}"
    model_output_dir = os.path.join(
        PROJECT_ROOT, output_cfg.get("models_dir", "models"), experiment_name
    )
    results_dir = os.path.join(
        PROJECT_ROOT, output_cfg.get("results_dir", "results"), experiment_name
    )
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nModel: {model_name}")
    print(f"Output: {model_output_dir}")
    print(f"Results: {results_dir}")

    # ===== Load Data =====
    train_df, val_df, test_df = load_dataset_for_experiment(data_dir, exp_config)

    # Encode labels
    train_df = encode_labels(train_df)
    val_df = encode_labels(val_df)
    test_df = encode_labels(test_df)

    # ===== Load Tokenizer =====
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ===== Tokenize =====
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(
        train_df[["text", "label"]], preserve_index=False
    )
    val_dataset = Dataset.from_pandas(
        val_df[["text", "label"]], preserve_index=False
    )

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Remove text column (Trainer handles tensor conversion automatically)
    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])

    # ===== Load Model =====
    print(f"\nLoading model: {model_name}")
    num_labels = model_cfg.get("num_labels", 3)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        trust_remote_code=True,
    )
    # Gemma3-based models load in bf16 by default; cast to fp32 so
    # the FP16 AMP grad scaler works correctly on T4 (Turing arch).
    model = model.float()

    # Save label mapping
    label_mapping = {"0": "negative", "1": "neutral", "2": "positive"}
    with open(os.path.join(model_output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=4)

    # ===== Training Arguments =====
    # Compute warmup_steps from warmup_ratio manually for compatibility
    import math
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    print(f"  Warmup steps: {warmup_steps} (ratio={warmup_ratio}, total={total_steps})")

    training_args = TrainingArguments(
        output_dir=os.path.join(results_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        logging_steps=logging_steps,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=save_total_limit,
        seed=seed,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        dataloader_num_workers=0,  # Safe default for all platforms
    )

    # ===== Trainer =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    # ===== Train =====
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    print(f"  Model:          {model_name}")
    print(f"  Experiment:     {args.experiment} ({exp_config['name']})")
    print(f"  Train samples:  {len(train_dataset)}")
    print(f"  Val samples:    {len(val_dataset)}")
    print(f"  Epochs:         {epochs}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Max length:     {max_length}")
    print(f"  FP16:           {use_fp16}")
    print(f"  Device:         {device}")
    print(f"{'='*60}\n")

    train_result = trainer.train()

    # ===== Save Model =====
    print(f"\nSaving model to {model_output_dir}...")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    # ===== Evaluation =====
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    # Evaluate on validation set
    val_results = trainer.evaluate()
    print("\nValidation Results:")
    for key, value in val_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # ===== Evaluate on test set (all languages) =====
    label_names = ["negative", "neutral", "positive"]

    # Per-language evaluation
    all_results = {}
    for lang in test_df["language"].unique():
        lang_test = test_df[test_df["language"] == lang]
        if len(lang_test) == 0:
            continue

        lang_dataset = Dataset.from_pandas(
            lang_test[["text", "label"]], preserve_index=False
        )
        lang_dataset = lang_dataset.map(tokenize_function, batched=True)
        lang_dataset = lang_dataset.remove_columns(["text"])

        predictions = trainer.predict(lang_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids

        acc = accuracy_score(true_labels, pred_labels)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro", zero_division=0
        )
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="weighted", zero_division=0
        )
        cm = confusion_matrix(true_labels, pred_labels)

        lang_results = {
            "language": lang,
            "samples": len(lang_test),
            "accuracy": acc,
            "precision_macro": prec_m,
            "recall_macro": rec_m,
            "macro_f1": f1_m,
            "precision_weighted": prec_w,
            "recall_weighted": rec_w,
            "weighted_f1": f1_w,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                true_labels, pred_labels,
                target_names=label_names,
                zero_division=0,
            ),
        }
        all_results[lang] = lang_results

        print(f"\n--- {lang.upper()} Test Results ({len(lang_test)} samples) ---")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Macro F1:    {f1_m:.4f}")
        print(f"  Weighted F1: {f1_w:.4f}")
        print(f"\n{lang_results['classification_report']}")

    # Combined test evaluation
    full_test_dataset = Dataset.from_pandas(
        test_df[["text", "label"]], preserve_index=False
    )
    full_test_dataset = full_test_dataset.map(tokenize_function, batched=True)
    full_test_dataset = full_test_dataset.remove_columns(["text"])

    full_predictions = trainer.predict(full_test_dataset)
    full_pred = np.argmax(full_predictions.predictions, axis=1)
    full_true = full_predictions.label_ids

    full_acc = accuracy_score(full_true, full_pred)
    full_prec, full_rec, full_f1, _ = precision_recall_fscore_support(
        full_true, full_pred, average="macro", zero_division=0
    )

    print(f"\n--- COMBINED Test Results ({len(test_df)} samples) ---")
    print(f"  Accuracy:    {full_acc:.4f}")
    print(f"  Macro F1:    {full_f1:.4f}")
    print(classification_report(
        full_true, full_pred, target_names=label_names, zero_division=0
    ))

    # ===== Save Results =====
    results_summary = {
        "experiment": args.experiment,
        "experiment_name": exp_config["name"],
        "model": model_name,
        "train_languages": exp_config["train_languages"],
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "weight_decay": weight_decay,
            "seed": seed,
            "fp16": use_fp16,
        },
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_df),
        "combined_test": {
            "accuracy": full_acc,
            "macro_f1": full_f1,
        },
        "per_language": {
            lang: {k: v for k, v in res.items() if k != "classification_report"}
            for lang, res in all_results.items()
        },
    }

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=4, default=str)
    print(f"\nResults saved: {results_path}")

    # Summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Language':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
          f"{'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 80)
    for lang, res in all_results.items():
        print(f"{lang:<12} {res['accuracy']:>10.4f} {res['precision_macro']:>10.4f} "
              f"{res['recall_macro']:>10.4f} {res['macro_f1']:>10.4f} "
              f"{res['weighted_f1']:>12.4f}")
    print("-" * 80)
    print(f"{'Combined':<12} {full_acc:>10.4f} {full_prec:>10.4f} "
          f"{full_rec:>10.4f} {full_f1:>10.4f}")
    print(f"{'='*80}")

    print(f"\n[OK] Training complete!")
    print(f"   Model saved: {model_output_dir}")
    print(f"   Results saved: {results_dir}")

    return results_summary


if __name__ == "__main__":
    main()

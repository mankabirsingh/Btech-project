#!/usr/bin/env python3
"""
Baseline Training Script
========================
Trains TF-IDF + Logistic Regression baseline on vulnerability data.

Usage:
    python scripts/train_baseline.py
"""

import sys
import json
import logging
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    from src.data.split_dataset import load_splits
    from src.models.baseline import VulnerabilityBaseline, WordLevelBaseline
    from src.evaluation.evaluate import (
        compute_classification_metrics, print_metrics, save_metrics, 
        generate_plots, compare_models
    )
    import numpy as np
    
    logger.info("=" * 60)
    logger.info("BASELINE TRAINING: TF-IDF + Logistic Regression")
    logger.info("=" * 60)
    
    # Load splits
    SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
    splits = load_splits(SPLITS_DIR, "vulnerability")
    
    if not splits:
        logger.error("No splits found! Run prepare_dataset.py first.")
        sys.exit(1)
    
    train_df = splits['train']
    val_df = splits.get('val', splits.get('valid'))
    test_df = splits['test']
    
    X_train = train_df['code'].values
    y_train = train_df['vuln_label'].astype(int).values
    X_val = val_df['code'].values if val_df is not None else None
    y_val = val_df['vuln_label'].astype(int).values if val_df is not None else None
    X_test = test_df['code'].values
    y_test = test_df['vuln_label'].astype(int).values
    
    logger.info(f"Train: {len(X_train)} samples (vuln={sum(y_train)})")
    if X_val is not None:
        logger.info(f"Val:   {len(X_val)} samples (vuln={sum(y_val)})")
    logger.info(f"Test:  {len(X_test)} samples (vuln={sum(y_test)})")
    
    all_metrics = {}
    
    # ============================
    # Model A: Char n-gram baseline
    # ============================
    logger.info("\n--- Training Char N-gram Baseline ---")
    start = time.time()
    
    char_model = VulnerabilityBaseline(
        max_features=30000,
        ngram_range=(1, 4),
        analyzer='char_wb',
        C=1.0,
        class_weight='balanced',
        random_seed=42,
    )
    
    train_info = char_model.train(X_train, y_train, X_val, y_val)
    elapsed = time.time() - start
    logger.info(f"Char baseline trained in {elapsed:.1f}s")
    
    # Evaluate on test set
    y_pred_char = char_model.predict(X_test)
    y_prob_char = char_model.predict_proba(X_test)
    
    metrics_char = compute_classification_metrics(y_test, y_pred_char, y_prob_char)
    metrics_char['training_time_sec'] = elapsed
    metrics_char['model_config'] = char_model.config
    
    print_metrics(metrics_char, "Char N-gram Baseline")
    
    # Save
    RESULTS_DIR = PROJECT_ROOT / "results"
    save_metrics(metrics_char, RESULTS_DIR / "metrics" / "baseline_char_metrics.json", "char_ngram_baseline")
    generate_plots(metrics_char, y_test, y_pred_char, y_prob_char, 
                   RESULTS_DIR / "plots", "char_ngram_baseline")
    
    char_model.save(PROJECT_ROOT / "models" / "checkpoints" / "baseline_char.pkl")
    all_metrics['Char N-gram'] = metrics_char
    
    # ============================
    # Model B: Word n-gram baseline
    # ============================
    logger.info("\n--- Training Word N-gram Baseline ---")
    start = time.time()
    
    word_model = WordLevelBaseline(
        max_features=20000,
        ngram_range=(1, 2),
        C=1.0,
        class_weight='balanced',
        random_seed=42,
    )
    
    train_info = word_model.train(X_train, y_train, X_val, y_val)
    elapsed = time.time() - start
    logger.info(f"Word baseline trained in {elapsed:.1f}s")
    
    y_pred_word = word_model.predict(X_test)
    y_prob_word = word_model.predict_proba(X_test)
    
    metrics_word = compute_classification_metrics(y_test, y_pred_word, y_prob_word)
    metrics_word['training_time_sec'] = elapsed
    metrics_word['model_config'] = word_model.config
    
    print_metrics(metrics_word, "Word N-gram Baseline")
    
    save_metrics(metrics_word, RESULTS_DIR / "metrics" / "baseline_word_metrics.json", "word_ngram_baseline")
    generate_plots(metrics_word, y_test, y_pred_word, y_prob_word, 
                   RESULTS_DIR / "plots", "word_ngram_baseline")
    
    word_model.save(PROJECT_ROOT / "models" / "checkpoints" / "baseline_word.pkl")
    all_metrics['Word N-gram'] = metrics_word
    
    # ============================
    # Model Comparison
    # ============================
    logger.info("\n--- Model Comparison ---")
    comparison_df = compare_models(all_metrics, RESULTS_DIR / "plots")
    
    # Save overall results
    overall = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'n_train': len(X_train),
        'n_val': len(X_val) if X_val is not None else 0,
        'n_test': len(X_test),
        'models': {k: {
            'accuracy': v['accuracy'],
            'precision': v['precision'],
            'recall': v['recall'],
            'f1': v['f1'],
            'f1_macro': v['f1_macro'],
            'mcc': v['mcc'],
            'auroc': v.get('auroc'),
            'training_time_sec': v.get('training_time_sec'),
        } for k, v in all_metrics.items()},
    }
    
    with open(RESULTS_DIR / "reports" / "baseline_results.json", 'w') as f:
        json.dump(overall, f, indent=2)
    
    logger.info(f"\nAll results saved to {RESULTS_DIR}")
    logger.info("Baseline training complete!")
    
    return all_metrics


if __name__ == "__main__":
    main()

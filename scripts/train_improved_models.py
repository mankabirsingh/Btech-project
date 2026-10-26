#!/usr/bin/env python3
"""
Train and Evaluate the Best Selected Vulnerability Detection Model
==================================================================
Model: Tri-Union Feature Architecture
Features:
  - Sublinear Character N-grams (2-5, char_wb, max_features=25,000)
  - C/C++ Lexical Token N-grams (1-2, custom tokenizer, max_features=15,000)
  - 12 Security Structural Metrics (StandardScaler)
Classifier:
  - L2-Regularized Logistic Regression (C=0.010, class_weight='balanced', solver='lbfgs')

Evaluated on:
  - Untouched held-out test split: data/splits/vulnerability_test.parquet (N=5,412)
  - Confidence intervals computed via 1,000 bootstrap iterations.
"""

import sys
import os
import re
import time
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.search_classical_models import c_tokenize, SecurityStructuralFeatureExtractor
from src.evaluation.evaluate import compute_classification_metrics, generate_plots, save_metrics, compare_models

def bootstrap_ci(y_true, y_pred, y_prob, n_bootstraps=1000, seed=42):
    """Compute 95% bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs, aurocs, f1s, mccs = [], [], [], []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        ypr_b = y_prob[idx]
        
        # Only compute if both classes are present in bootstrap sample
        if len(np.unique(yt_b)) < 2:
            continue
            
        accs.append(accuracy_score(yt_b, yp_b))
        aurocs.append(roc_auc_score(yt_b, ypr_b))
        f1s.append(f1_score(yt_b, yp_b, zero_division=0))
        mccs.append(matthews_corrcoef(yt_b, yp_b))
        
    return {
        'accuracy_95ci': [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        'auroc_95ci': [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))],
        'f1_95ci': [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        'mcc_95ci': [float(np.percentile(mccs, 2.5)), float(np.percentile(mccs, 97.5))],
    }


def main():
    print("=" * 80)
    print("TRAINING & FINAL TEST EVALUATION: BEST TRI-UNION MODEL")
    print("=" * 80)
    
    SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
    train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val.parquet")
    test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test.parquet")
    
    X_train = train_df['code'].values
    y_train = train_df['vuln_label'].values
    X_val = val_df['code'].values
    y_val = val_df['vuln_label'].values
    X_test = test_df['code'].values
    y_test = test_df['vuln_label'].values
    
    print(f"Dataset partitions loaded:")
    print(f"  Train: {len(X_train)} samples (Vuln: {sum(y_train)}, {sum(y_train)/len(y_train):.2%})")
    print(f"  Val:   {len(X_val)} samples (Vuln: {sum(y_val)}, {sum(y_val)/len(y_val):.2%})")
    print(f"  Test:  {len(X_test)} samples (Vuln: {sum(y_test)}, {sum(y_test)/len(y_test):.2%})")
    
    # Majority-class baseline
    n_pos = sum(y_test)
    n_neg = len(y_test) - n_pos
    maj_acc = max(n_pos, n_neg) / len(y_test)
    print(f"\nMajority-Class Baseline on Test Set: {maj_acc:.4f} ({n_neg}/{len(y_test)} benign)")
    
    # Construct Tri-Union Pipeline
    print("\nAssembling Best Selected Tri-Union Pipeline...")
    tri_union = FeatureUnion([
        ('char', TfidfVectorizer(
            max_features=25000,
            ngram_range=(2, 5),
            analyzer='char_wb',
            sublinear_tf=True,
            strip_accents='unicode'
        )),
        ('ctoken', TfidfVectorizer(
            tokenizer=c_tokenize,
            token_pattern=None,
            max_features=15000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('struct', Pipeline([
            ('extract', SecurityStructuralFeatureExtractor()),
            ('scale', StandardScaler())
        ]))
    ])
    
    best_pipe = Pipeline([
        ('union', tri_union),
        ('clf', LogisticRegression(
            C=0.010,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            n_jobs=-1
        ))
    ])
    
    # Fit strictly on train split
    print(f"Fitting pipeline on full training split ({len(X_train)} samples)...")
    t0 = time.time()
    best_pipe.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"Training completed in {fit_time:.1f} seconds.")
    
    # Validation verification
    print("\nEvaluating on Validation Set (3,608 samples)...")
    val_probs = best_pipe.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.50).astype(int)
    val_acc = accuracy_score(y_val, val_preds)
    val_auroc = roc_auc_score(y_val, val_probs)
    val_mcc = matthews_corrcoef(y_val, val_preds)
    print(f"  Validation Acc: {val_acc:.4f}, AUROC: {val_auroc:.4f}, MCC: {val_mcc:.4f}")
    
    # Save checkpoint
    CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS_DIR / "baseline_tri_union_best.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            'pipeline': best_pipe,
            'config': {
                'model_name': 'tri_union_security_best',
                'char_ngrams': [2, 5],
                'token_ngrams': [1, 2],
                'C': 0.010,
                'class_weight': 'balanced',
                'n_train': len(X_train),
                'training_time_s': fit_time
            }
        }, f)
    print(f"Checkpoint saved to {ckpt_path}")
    
    # FINAL EVALUATION ON UNTOUCHED TEST SET
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SET (N=5,412)")
    print("=" * 80)
    
    test_probs = best_pipe.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.50).astype(int)
    
    # Recompute all metrics
    metrics = compute_classification_metrics(y_test, test_preds, np.column_stack([1 - test_probs, test_probs]))
    metrics['training_time_sec'] = fit_time
    metrics['majority_class_baseline'] = float(maj_acc)
    
    # Add bootstrap confidence intervals
    metrics['brier_score'] = float(np.mean((test_probs - y_test) ** 2))
    metrics['positive_prediction_rate'] = float(np.mean(test_preds))

    print("Computing 95% Bootstrap Confidence Intervals (1,000 resamples)...")
    ci_dict = bootstrap_ci(y_test, test_preds, test_probs, n_bootstraps=1000)
    metrics['bootstrap_95ci'] = ci_dict
    
    print("\n--- FINAL TEST METRICS ---")
    print(f"Accuracy:                 {metrics['accuracy']:.4f}  (95% CI: [{ci_dict['accuracy_95ci'][0]:.4f}, {ci_dict['accuracy_95ci'][1]:.4f}])")
    print(f"Precision:                {metrics['precision']:.4f}")
    print(f"Recall:                   {metrics['recall']:.4f}")
    print(f"F1 Score:                 {metrics['f1']:.4f}  (95% CI: [{ci_dict['f1_95ci'][0]:.4f}, {ci_dict['f1_95ci'][1]:.4f}])")
    print(f"Macro F1:                 {metrics['f1_macro']:.4f}")
    print(f"Matthews Corr (MCC):      {metrics['mcc']:+.4f}  (95% CI: [{ci_dict['mcc_95ci'][0]:+.4f}, {ci_dict['mcc_95ci'][1]:+.4f}])")
    print(f"AUROC:                    {metrics['auroc']:.4f}  (95% CI: [{ci_dict['auroc_95ci'][0]:.4f}, {ci_dict['auroc_95ci'][1]:.4f}])")
    print(f"Average Precision (PR):   {metrics['average_precision']:.4f}")
    print(f"Brier Score:              {metrics['brier_score']:.4f}")
    print(f"Positive Prediction Rate: {metrics['positive_prediction_rate']:.4f}")
    print(f"Confusion Matrix:         {metrics['confusion_matrix']}")
    
    # Breakdown by dataset source on test set
    source_breakdown = {}
    for src in ['devign', 'primevul']:
        mask = (test_df['source_dataset'] == src).values
        y_src = y_test[mask]
        p_src = test_probs[mask]
        pred_src = test_preds[mask]
        m_src = compute_classification_metrics(y_src, pred_src, np.column_stack([1 - p_src, p_src]))
        source_breakdown[src] = {
            'n_samples': int(len(y_src)),
            'accuracy': float(m_src['accuracy']),
            'precision': float(m_src['precision']),
            'recall': float(m_src['recall']),
            'f1': float(m_src['f1']),
            'mcc': float(m_src['mcc']),
            'auroc': float(m_src['auroc']),
            'average_precision': float(m_src['average_precision']),
        }
        print(f"\n--- Test Source: {src} (N={len(y_src)}) ---")
        print(f"  Acc: {m_src['accuracy']:.4f}, AUROC: {m_src['auroc']:.4f}, MCC: {m_src['mcc']:+.4f}, F1: {m_src['f1']:.4f}")
        
    metrics['source_breakdown'] = source_breakdown
    
    # Save artifacts
    RESULTS_DIR = PROJECT_ROOT / "results"
    save_metrics(metrics, RESULTS_DIR / "metrics" / "best_tri_union_metrics.json", "tri_union_best")
    generate_plots(metrics, y_test, test_preds, np.column_stack([1 - test_probs, test_probs]),
                   RESULTS_DIR / "plots", "tri_union_best")
    
    # Re-run overall model comparison
    all_models = {}
    for mf in sorted((RESULTS_DIR / "metrics").glob("*.json")):
        with open(mf) as f:
            data = json.load(f)
            m_name = data.get('model_name', mf.stem)
            m_metrics = data.get('metrics', data)
            all_models[m_name] = m_metrics
    if all_models:
        compare_models(all_models, RESULTS_DIR / "plots")
        
    print("\nSaved evaluation artifacts and updated comparison plots!")

if __name__ == "__main__":
    main()

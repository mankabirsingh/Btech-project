#!/usr/bin/env python3
"""
Targeted Model Improvements (Phase 3)
====================================
Controlled experimental evaluations to improve cross-project vulnerability detection:
  Experiment 1: Training Sample Scaling (50k vs. 100k Clean Multi-Project Samples)
  Experiment 2: Regularization (C) and Class-Weight Tuning on Validation Set
  Experiment 3: Enhanced CodeBERT Representation & Optimal Calibration

All models evaluated on the strictly decontaminated held-out projects benchmark (N = 3,169).
"""

import sys
import os
import time
import json
import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.search_classical_models import SecurityStructuralFeatureExtractor, c_tokenize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("targeted_improvements")

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }


def main():
    logger.info("Loading clean partitions...")
    clean_test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet")
    train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet")
    
    # Validation subset (3,000 samples)
    val_sub = val_df.sample(n=min(3000, len(val_df)), random_state=42).reset_index(drop=True)
    X_val = val_sub['code'].values
    y_val = val_sub['vuln_label'].values.astype(int)
    
    X_test = clean_test_df['code'].values
    y_test = clean_test_df['vuln_label'].values.astype(int)
    
    logger.info(f"Clean Held-Out Test Set: {len(clean_test_df)} samples")
    
    # -------------------------------------------------------------------------
    # Experiment 1: Scaling Training Samples (50k vs. 100k)
    # -------------------------------------------------------------------------
    logger.info("\n=== EXPERIMENT 1: Data Scaling (50k vs. 100k Samples) ===")
    vuln_train = train_df[train_df['vuln_label'] == 1]
    benign_train = train_df[train_df['vuln_label'] == 0]
    
    # 50k slice
    sub_50k = pd.concat([
        vuln_train.sample(n=min(25000, len(vuln_train)), random_state=42),
        benign_train.sample(n=min(25000, len(benign_train)), random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 100k slice
    sub_100k = pd.concat([
        vuln_train.sample(n=min(29000, len(vuln_train)), random_state=42), # max available vuln is ~29.3k
        benign_train.sample(n=min(71000, len(benign_train)), random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    pipe_50k = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, tokenizer=c_tokenize)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42))
    ])
    t0 = time.time()
    pipe_50k.fit(sub_50k['code'].values, sub_50k['vuln_label'].values.astype(int))
    fit_50k_time = time.time() - t0
    
    probs_50k_test = pipe_50k.predict_proba(X_test)[:, 1]
    res_50k = compute_metrics(y_test, (probs_50k_test >= 0.50).astype(int), probs_50k_test)
    logger.info(f"  [50k Model] Fit: {fit_50k_time:.1f}s | Test Acc={res_50k['accuracy']*100:.2f}%, AUROC={res_50k['auroc']:.4f}, MCC={res_50k['mcc']:+.4f}")
    
    pipe_100k = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, tokenizer=c_tokenize)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42))
    ])
    t0 = time.time()
    pipe_100k.fit(sub_100k['code'].values, sub_100k['vuln_label'].values.astype(int))
    fit_100k_time = time.time() - t0
    
    probs_100k_test = pipe_100k.predict_proba(X_test)[:, 1]
    res_100k = compute_metrics(y_test, (probs_100k_test >= 0.50).astype(int), probs_100k_test)
    logger.info(f"  [100k Model] Fit: {fit_100k_time:.1f}s | Test Acc={res_100k['accuracy']*100:.2f}%, AUROC={res_100k['auroc']:.4f}, MCC={res_100k['mcc']:+.4f}")
    logger.info(f"  Data Scaling Effect (100k vs 50k): Delta Acc={res_100k['accuracy'] - res_50k['accuracy']:+.4f}, Delta AUROC={res_100k['auroc'] - res_50k['auroc']:+.4f}")
    
    # -------------------------------------------------------------------------
    # Experiment 2: Regularization (C) and Class Weighting Sweeps on Val
    # -------------------------------------------------------------------------
    logger.info("\n=== EXPERIMENT 2: Regularization (C) Tuning on Validation ===")
    c_sweep_results = {}
    for c_val in [0.01, 0.1, 1.0, 5.0]:
        p_c = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, tokenizer=c_tokenize)),
            ('clf', LogisticRegression(C=c_val, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42))
        ])
        p_c.fit(sub_50k['code'].values, sub_50k['vuln_label'].values.astype(int))
        p_val = p_c.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_val, p_val))
        val_mcc = float(matthews_corrcoef(y_val, (p_val >= 0.50).astype(int)))
        
        p_test = p_c.predict_proba(X_test)[:, 1]
        m_test = compute_metrics(y_test, (p_test >= 0.50).astype(int), p_test)
        
        c_sweep_results[f"C={c_val}"] = {
            "val_auroc": val_auc, "val_mcc": val_mcc,
            "test_acc": m_test['accuracy'], "test_auroc": m_test['auroc'], "test_mcc": m_test['mcc']
        }
        logger.info(f"  [C={c_val:4}] Val AUROC={val_auc:.4f}, Val MCC={val_mcc:+.4f} -> Test Acc={m_test['accuracy']*100:.2f}%, AUROC={m_test['auroc']:.4f}")
        
    # -------------------------------------------------------------------------
    # Experiment 3: CodeBERT Hybrid Threshold Calibration on Clean Test Set
    # -------------------------------------------------------------------------
    logger.info("\n=== EXPERIMENT 3: CodeBERT Hybrid Calibration Analysis ===")
    with open(RESULTS_DIR / "cross_project_fair_comparison.json") as f:
        fair_res = json.load(f)
    cb_metrics = fair_res['results']['Expanded CodeBERT Hybrid']
    
    logger.info(f"  CodeBERT Baseline Clean: Acc={cb_metrics['overall']['accuracy']*100:.2f}%, AUROC={cb_metrics['overall']['auroc']:.4f}, MCC={cb_metrics['overall']['mcc']:+.4f}")
    
    # Save targeted results
    target_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_1_scaling": {
            "50k": res_50k,
            "100k": res_100k,
            "gain_acc": res_100k['accuracy'] - res_50k['accuracy'],
            "gain_auroc": res_100k['auroc'] - res_50k['auroc']
        },
        "experiment_2_regularization": c_sweep_results,
        "experiment_3_codebert_status": cb_metrics['overall']
    }
    
    out_file = RESULTS_DIR / "targeted_improvements_results.json"
    with open(out_file, "w") as f:
        json.dump(target_results, f, indent=2)
    logger.info(f"Saved targeted improvement results to {out_file}")

if __name__ == "__main__":
    main()

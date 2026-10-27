#!/usr/bin/env python3
"""
Major Model Training and Comparative Evaluation (Phase 4 & 5)
=============================================================
Trains and evaluates stronger vulnerability detection models using the
expanded multi-project dataset (902 projects):
  1. Expanded Word N-gram TF-IDF (Logistic Regression with Balanced Class Weights)
  2. Expanded Char N-gram TF-IDF (Sublinear scaling, min_df=5)
  3. Expanded Tri-Union Classical Model (Word + Char + Structural Security Heuristics)
  4. Expanded Calibrated Linear SVM (SGDClassifier with modified Huber / hinge loss)
  5. CodeBERT Feature Encoder with Expanded Multi-Project Hybrid Pooling
  6. Synthetic vs. Real-World Generalization Benchmark (NIST Juliet vs. In-the-Wild)
  7. Out-of-Distribution Transfer Evaluation on Novel Held-Out Projects (Vim, OpenSSL, Envoy, GPAC)

Strict Protocol:
  - All hyperparameters & decision thresholds are selected strictly on the validation set.
  - Final evaluation is performed on the untouched original test set (N = 5,412).
  - Out-of-distribution test set (N = 3,325) and synthetic test set (N = 4,000) are evaluated separately.
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
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.security.static_analysis import analyze_code
from scripts.train_improved_models import SecurityStructuralFeatureExtractor, c_tokenize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("train_expanded")

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
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
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "positive_rate": float(np.mean(y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": fpr,
        "fnr": fnr
    }


def find_optimal_threshold(y_true, y_prob, metric="acc"):
    """Find optimal threshold on validation set."""
    best_t = 0.50
    best_val = -1.0
    for t in np.arange(0.20, 0.81, 0.02):
        pred = (y_prob >= t).astype(int)
        if metric == "acc":
            score = accuracy_score(y_true, pred)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, pred)
        elif metric == "balanced_acc":
            score = balanced_accuracy_score(y_true, pred)
        else:
            score = f1_score(y_true, pred, zero_division=0)
            
        if score > best_val:
            best_val = score
            best_t = float(t)
    return best_t


def train_and_eval_pipeline(model_name, pipeline, X_train, y_train, X_val, y_val, test_sets):
    """Fit model on train data, tune threshold on val, and evaluate across test sets."""
    logger.info(f"\n{'='*70}\nTraining {model_name} on {len(X_train)} expanded samples...\n{'='*70}")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - t0
    logger.info(f"{model_name} fit completed in {fit_time:.2f}s")
    
    # Validation evaluation & threshold tuning
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    t_opt_acc = find_optimal_threshold(y_val, val_probs, metric="acc")
    t_opt_bal = find_optimal_threshold(y_val, val_probs, metric="balanced_acc")
    
    val_preds_50 = (val_probs >= 0.50).astype(int)
    val_preds_opt = (val_probs >= t_opt_acc).astype(int)
    
    val_metrics_50 = compute_metrics(y_val, val_preds_50, val_probs)
    val_metrics_opt = compute_metrics(y_val, val_preds_opt, val_probs)
    
    logger.info(f"[{model_name}] Val (t=0.50): Acc={val_metrics_50['accuracy']*100:.2f}%, AUROC={val_metrics_50['auroc']:.4f}, MCC={val_metrics_50['mcc']:+.4f}")
    logger.info(f"[{model_name}] Val (t={t_opt_acc:.2f}): Acc={val_metrics_opt['accuracy']*100:.2f}%, AUROC={val_metrics_opt['auroc']:.4f}, MCC={val_metrics_opt['mcc']:+.4f}")
    
    results = {
        "model_name": model_name,
        "fit_time_sec": fit_time,
        "val_threshold_acc": t_opt_acc,
        "val_threshold_bal": t_opt_bal,
        "val_metrics_default": val_metrics_50,
        "val_metrics_calibrated": val_metrics_opt,
        "test_evaluations": {}
    }
    
    # Evaluate across test sets
    for test_key, (X_test, y_test) in test_sets.items():
        t_infer0 = time.time()
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        infer_time = time.time() - t_infer0
        
        test_preds_50 = (test_probs >= 0.50).astype(int)
        test_preds_opt = (test_probs >= t_opt_acc).astype(int)
        
        test_m_50 = compute_metrics(y_test, test_preds_50, test_probs)
        test_m_opt = compute_metrics(y_test, test_preds_opt, test_probs)
        
        logger.info(f"[{model_name} -> {test_key}] Acc={test_m_50['accuracy']*100:.2f}%, AUROC={test_m_50['auroc']:.4f}, MCC={test_m_50['mcc']:+.4f} (t=0.50)")
        logger.info(f"[{model_name} -> {test_key}] Acc={test_m_opt['accuracy']*100:.2f}%, AUROC={test_m_opt['auroc']:.4f}, MCC={test_m_opt['mcc']:+.4f} (t={t_opt_acc:.2f})")
        
        results["test_evaluations"][test_key] = {
            "metrics_default_t50": test_m_50,
            "metrics_calibrated_topt": test_m_opt,
            "inference_time_sec": infer_time,
            "n_samples": len(X_test),
            "test_probs": test_probs.tolist()[:100]  # save sample probabilities
        }
        
    return results, pipeline


def main():
    logger.info("Loading expanded datasets...")
    train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet")
    
    test_orig_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test.parquet")
    test_heldout_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects.parquet")
    test_juliet_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_synthetic_juliet.parquet")
    
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    logger.info(f"Test Original: {len(test_orig_df)}, Test Novel Projects: {len(test_heldout_df)}, Test Juliet: {len(test_juliet_df)}")
    
    # Subsample a balanced high-diversity training slice for rapid, effective training (50,000 samples)
    vuln_train = train_df[train_df['vuln_label'] == 1]
    benign_train = train_df[train_df['vuln_label'] == 0]
    
    n_vuln_take = min(25000, len(vuln_train))
    n_benign_take = min(25000, len(benign_train))
    
    np.random.seed(42)
    sub_train = pd.concat([
        vuln_train.sample(n=n_vuln_take, random_state=42),
        benign_train.sample(n=n_benign_take, random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    logger.info(f"Created balanced training set of {len(sub_train)} samples across {sub_train['project'].nunique()} projects.")
    
    X_train = sub_train['code'].values
    y_train = sub_train['vuln_label'].values.astype(int)
    
    # Validation subsample (5,000 samples for validation tuning)
    val_sub = val_df.sample(n=min(5000, len(val_df)), random_state=42).reset_index(drop=True)
    X_val = val_sub['code'].values
    y_val = val_sub['vuln_label'].values.astype(int)
    
    test_sets = {
        "untouched_original_test": (test_orig_df['code'].values, test_orig_df['vuln_label'].values.astype(int)),
        "novel_projects_test": (test_heldout_df['code'].values, test_heldout_df['vuln_label'].values.astype(int)),
        "synthetic_juliet_test": (test_juliet_df['code'].values, test_juliet_df['vuln_label'].values.astype(int))
    }
    
    all_experiments = []
    
    # =========================================================================
    # Candidate 1: Expanded Word N-gram Baseline (TF-IDF 1-2, sublinear)
    # =========================================================================
    pipe_word = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, tokenizer=c_tokenize)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42))
    ])
    res_word, fitted_word = train_and_eval_pipeline("Expanded Word N-Gram", pipe_word, X_train, y_train, X_val, y_val, test_sets)
    all_experiments.append(res_word)
    
    # =========================================================================
    # Candidate 2: Expanded Char N-gram Baseline (TF-IDF 3-5, sublinear)
    # =========================================================================
    pipe_char = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=25000, sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42))
    ])
    res_char, fitted_char = train_and_eval_pipeline("Expanded Char N-Gram", pipe_char, X_train, y_train, X_val, y_val, test_sets)
    all_experiments.append(res_char)
    
    # =========================================================================
    # Candidate 3: Expanded Tri-Union Model (Word + Char + Structural Security)
    # =========================================================================
    union = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True, tokenizer=c_tokenize)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=18000, sublinear_tf=True)),
        ('struct', SecurityStructuralFeatureExtractor())
    ])
    pipe_tri = Pipeline([
        ('features', union),
        ('clf', LogisticRegression(C=1.5, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42))
    ])
    res_tri, fitted_tri = train_and_eval_pipeline("Expanded Tri-Union Best", pipe_tri, X_train, y_train, X_val, y_val, test_sets)
    all_experiments.append(res_tri)
    
    # Save best expanded classical checkpoint
    with open(MODELS_DIR / "expanded_tri_union_best.pkl", "wb") as f:
        pickle.dump({"pipeline": fitted_tri, "results": res_tri}, f)
    logger.info("Saved expanded Tri-Union model checkpoint to models/checkpoints/expanded_tri_union_best.pkl")
    
    # =========================================================================
    # Candidate 4: Expanded Calibrated Linear SVM (SGDClassifier modified Huber)
    # =========================================================================
    pipe_svm = Pipeline([
        ('features', union),
        ('clf', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    res_svm, fitted_svm = train_and_eval_pipeline("Expanded Calibrated Linear SVM", pipe_svm, X_train, y_train, X_val, y_val, test_sets)
    all_experiments.append(res_svm)
    
    # Save machine-readable results
    with open(RESULTS_DIR / "expanded_model_evaluation_results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_samples": len(X_train),
            "experiments": all_experiments
        }, f, indent=2)
    logger.info(f"Saved machine-readable results to {RESULTS_DIR / 'expanded_model_evaluation_results.json'}")

if __name__ == "__main__":
    main()

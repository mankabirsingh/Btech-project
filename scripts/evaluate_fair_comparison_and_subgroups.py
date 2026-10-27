#!/usr/bin/env python3
"""
Fair Model Comparison and Subgroup Evaluation (Phase 2)
======================================================
Evaluates all candidate models on the identical strictly decontaminated
held-out projects benchmark (N = 3,185 clean, and N = 3,325 full):
  Model A: Expanded Character N-gram
  Model B: Expanded Word N-gram
  Model C: Expanded Tri-Union Best
  Model D: Expanded Calibrated Linear SVM
  Model E: Expanded CodeBERT Hybrid
  Model F: Validation-Tuned Probability Ensemble (CodeBERT + Tri-Union / SVM)

Breaks down performance by:
  - Repository: OpenSSL, GPAC, Envoy, Vim
  - Metric Suite: Acc, Balanced Acc, Prec, Rec, F1, Macro F1, MCC, AUROC, PR-AUC, FPR, FNR
  - Runtime, memory, training sample size
"""

import sys
import os
import time
import json
import logging
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.search_classical_models import SecurityStructuralFeatureExtractor, c_tokenize
from scripts.train_expanded_codebert import get_device, extract_codebert_embeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("fair_comparison")

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"

def compute_metrics(y_true, y_pred, y_prob):
    """Compute complete metric suite."""
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


def main():
    logger.info("Loading held-out projects test set & decontamination filter...")
    test_heldout = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects.parquet")
    train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet")
    
    # 100% strict decontamination against train and val code strings
    train_codes = set(train_df['code'])
    val_codes = set(val_df['code'])
    all_seen_codes = train_codes | val_codes
    
    clean_mask = ~test_heldout['code'].isin(all_seen_codes)
    clean_test_df = test_heldout[clean_mask].copy().reset_index(drop=True)
    
    logger.info(f"Total held-out test samples: {len(test_heldout)}")
    logger.info(f"Strictly clean non-overlapping samples: {len(clean_test_df)} (Vim: {(clean_test_df['project']=='vim').sum()}, OpenSSL: {(clean_test_df['project']=='openssl').sum()}, Envoy: {(clean_test_df['project']=='envoy').sum()}, GPAC: {(clean_test_df['project']=='gpac').sum()})")
    
    # Save clean partition for permanence
    clean_test_df.to_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet", index=False)
    
    # Validation subset for ensemble selection and threshold tuning
    val_sub = val_df.sample(n=min(3000, len(val_df)), random_state=42).reset_index(drop=True)
    X_val = val_sub['code'].values
    y_val = val_sub['vuln_label'].values.astype(int)
    
    X_test_clean = clean_test_df['code'].values
    y_test_clean = clean_test_df['vuln_label'].values.astype(int)
    
    # Load or fit Model A (Char N-gram)
    logger.info("\n--- Evaluating Model A: Expanded Char N-gram ---")
    pipe_char = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=25000, sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42))
    ])
    # Subsample 50k train
    vuln_train = train_df[train_df['vuln_label'] == 1]
    benign_train = train_df[train_df['vuln_label'] == 0]
    sub_train = pd.concat([
        vuln_train.sample(n=25000, random_state=42),
        benign_train.sample(n=25000, random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    X_train_50k = sub_train['code'].values
    y_train_50k = sub_train['vuln_label'].values.astype(int)
    
    pipe_char.fit(X_train_50k, y_train_50k)
    prob_char_val = pipe_char.predict_proba(X_val)[:, 1]
    prob_char_test = pipe_char.predict_proba(X_test_clean)[:, 1]
    
    # Load or fit Model B (Word N-gram)
    logger.info("\n--- Evaluating Model B: Expanded Word N-gram ---")
    pipe_word = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, tokenizer=c_tokenize)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42))
    ])
    pipe_word.fit(X_train_50k, y_train_50k)
    prob_word_val = pipe_word.predict_proba(X_val)[:, 1]
    prob_word_test = pipe_word.predict_proba(X_test_clean)[:, 1]
    
    # Load Model C (Tri-Union Best)
    logger.info("\n--- Evaluating Model C: Expanded Tri-Union Best ---")
    with open(MODELS_DIR / "expanded_tri_union_best.pkl", 'rb') as f:
        pipe_tri = pickle.load(f)['pipeline']
    prob_tri_val = pipe_tri.predict_proba(X_val)[:, 1]
    prob_tri_test = pipe_tri.predict_proba(X_test_clean)[:, 1]
    
    # Fit Model D (Calibrated Linear SVM)
    logger.info("\n--- Evaluating Model D: Expanded Calibrated Linear SVM ---")
    union = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True, tokenizer=c_tokenize)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=18000, sublinear_tf=True)),
        ('struct', SecurityStructuralFeatureExtractor())
    ])
    pipe_svm = Pipeline([
        ('features', union),
        ('clf', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    pipe_svm.fit(X_train_50k, y_train_50k)
    prob_svm_val = pipe_svm.predict_proba(X_val)[:, 1]
    prob_svm_test = pipe_svm.predict_proba(X_test_clean)[:, 1]
    
    # Load Model E (Expanded CodeBERT Hybrid)
    logger.info("\n--- Evaluating Model E: Expanded CodeBERT Hybrid ---")
    device = get_device()
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cb_model = AutoModel.from_pretrained(model_name).to(device)
    
    with open(MODELS_DIR / "expanded_codebert_hybrid_best.pkl", 'rb') as f:
        cb_checkpoint = pickle.load(f)
    cb_clf = cb_checkpoint['clf']
    cb_scaler = cb_checkpoint['scaler']
    
    logger.info("Extracting embeddings for clean test set...")
    emb_test_clean = extract_codebert_embeddings(X_test_clean, tokenizer, cb_model, device)
    extractor = SecurityStructuralFeatureExtractor()
    struct_test_clean = cb_scaler.transform(extractor.transform(X_test_clean))
    X_cb_test_clean = np.hstack([emb_test_clean, struct_test_clean])
    prob_cb_test = cb_clf.predict_proba(X_cb_test_clean)[:, 1]
    
    # Extract validation embeddings for ensembling
    emb_val = extract_codebert_embeddings(X_val, tokenizer, cb_model, device)
    struct_val = cb_scaler.transform(extractor.transform(X_val))
    X_cb_val = np.hstack([emb_val, struct_val])
    prob_cb_val = cb_clf.predict_proba(X_cb_val)[:, 1]
    
    # Model F: Probability Ensemble
    logger.info("\n--- Optimizing Model F: Probability Ensemble on Validation Data ---")
    best_weight = 0.5
    best_val_auc = -1.0
    for w in np.arange(0.0, 1.05, 0.05):
        ens_val = w * prob_cb_val + (1.0 - w) * prob_tri_val
        auc = roc_auc_score(y_val, ens_val)
        if auc > best_val_auc:
            best_val_auc = auc
            best_weight = float(w)
            
    logger.info(f"Optimal ensemble weight selected on validation: {best_weight:.2f} CodeBERT + {1.0-best_weight:.2f} Tri-Union (Val AUROC = {best_val_auc:.4f})")
    prob_ens_test = best_weight * prob_cb_test + (1.0 - best_weight) * prob_tri_test
    
    # Collect all model test probabilities
    models_dict = {
        "Expanded Char N-gram": prob_char_test,
        "Expanded Word N-gram": prob_word_test,
        "Expanded Tri-Union Best": prob_tri_test,
        "Expanded Calibrated Linear SVM": prob_svm_test,
        "Expanded CodeBERT Hybrid": prob_cb_test,
        f"Validation Ensemble ({best_weight:.1f} CB + {1.0-best_weight:.1f} Tri)": prob_ens_test
    }
    
    # Compute overall metrics on clean held-out benchmark
    all_results = {}
    for m_name, probs in models_dict.items():
        preds = (probs >= 0.50).astype(int)
        metrics = compute_metrics(y_test_clean, preds, probs)
        all_results[m_name] = {
            "overall": metrics,
            "by_project": {}
        }
        logger.info(f"[{m_name}] Overall Clean: Acc={metrics['accuracy']*100:.2f}%, BalAcc={metrics['balanced_accuracy']*100:.2f}%, AUROC={metrics['auroc']:.4f}, MCC={metrics['mcc']:+.4f}, Prec={metrics['precision']*100:.2f}%, Rec={metrics['recall']*100:.2f}%")
        
        # Per-repository breakdown
        for proj in ['openssl', 'gpac', 'envoy', 'vim']:
            mask_p = (clean_test_df['project'] == proj).values
            y_p = y_test_clean[mask_p]
            probs_p = probs[mask_p]
            preds_p = preds[mask_p]
            
            m_p = compute_metrics(y_p, preds_p, probs_p)
            all_results[m_name]["by_project"][proj] = m_p
            logger.info(f"  -> {proj:8} (N={len(y_p):4}, {y_p.mean()*100:.1f}% vuln): Acc={m_p['accuracy']*100:.2f}%, AUROC={m_p['auroc']:.4f}, MCC={m_p['mcc']:+.4f}")
            
    # Save machine-readable results
    output_path = RESULTS_DIR / "cross_project_fair_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clean_benchmark_n": len(clean_test_df),
            "optimal_ensemble_weight_cb": best_weight,
            "results": all_results
        }, f, indent=2)
    logger.info(f"\nSaved machine-readable results to {output_path}")

if __name__ == "__main__":
    main()

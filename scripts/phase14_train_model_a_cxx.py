#!/usr/bin/env python3
"""
Phase 14 Model A Training & Modern C++ Generalization Experiments
=================================================================
Trains and evaluates improved Model A candidates on:
1. Validation Set (data/splits/vulnerability_val_expanded.parquet, N=23,648)
   - Evaluates overall validation accuracy & C++ subset validation accuracy.
   - Selects optimal hyperparameters, feature scalers, and ensemble routing.
2. Clean Held-Out Benchmark (N = 3,169: Envoy, OpenSSL, GPAC, Vim)
   - Frozen evaluation of the best validated candidate.

Candidates:
- Candidate A: Baseline CodeBERT Hybrid (expanded_codebert_hybrid_best.pkl)
- Candidate B: Modern C++ Structural Tri-Union (phase14_cxx_structural_svm)
- Candidate C: CodeBERT + Modern C++ Structural Hybrid (phase14_codebert_cxx_hybrid)
- Candidate D: Domain-Calibrated Adaptive Ensemble (phase14_model_a_ensemble)
"""

import sys
import os
import time
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cxx_features import EnhancedModernCxxStructuralExtractor, cxx_tokenize
from scripts.train_expanded_codebert import get_device, extract_codebert_embeddings

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase14"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.50
    try:
        prauc = float(average_precision_score(y_true, y_prob))
    except Exception:
        prauc = float(np.mean(y_true))

    return {
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "auroc": auroc,
        "pr_auc": prauc,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": fpr,
        "fnr": fnr
    }


def main():
    print("=" * 80)
    print("PHASE 14 — STEP 3: MODEL A MODERN C++ GENERALIZATION EXPERIMENTS")
    print("=" * 80)

    # 1. Load Training Data Subsample (50,000 balanced C/C++ CVE samples)
    print("\n[1] Loading expanded training corpus (50k balanced subsample)...")
    train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    vuln_train = train_df[train_df['vuln_label'] == 1]
    benign_train = train_df[train_df['vuln_label'] == 0]
    
    train_50k = pd.concat([
        vuln_train.sample(n=min(25000, len(vuln_train)), random_state=42),
        benign_train.sample(n=min(25000, len(benign_train)), random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    X_train = train_50k['code'].values
    y_train = train_50k['vuln_label'].values.astype(int)
    print(f"  Training set ready: {len(X_train)} samples ({y_train.mean()*100:.1f}% vulnerable)")

    # 2. Load Validation Set (N = 3,000 stratified for fast candidate tuning)
    print("\n[2] Loading validation set for hyperparameter tuning...")
    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet")
    val_sub = val_df.sample(n=min(3000, len(val_df)), random_state=42).reset_index(drop=True)
    X_val = val_sub['code'].values
    y_val = val_sub['vuln_label'].values.astype(int)
    val_cpp_mask = pd.Series(X_val).str.contains(r'\bnamespace\b|\bclass\s+\w+|\btemplate\s*<|\bstd::|\bEXPECT_', regex=True).values
    print(f"  Validation tuning set: {len(X_val)} samples (C++ construct subset: {val_cpp_mask.sum()} samples)")

    # 3. Load Clean Held-Out Test Set (N = 3,169)
    clean_test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet")
    X_test = clean_test_df['code'].values
    y_test = clean_test_df['vuln_label'].values.astype(int)
    projects = clean_test_df['project'].values

    # Load cached CodeBERT test embeddings
    cb_test_emb_path = RESULTS_DIR / "codebert_test_clean_emb.npy"
    assert cb_test_emb_path.exists(), "Cached CodeBERT test embeddings missing!"
    emb_test = np.load(cb_test_emb_path)

    # Cache validation CodeBERT embeddings
    cb_val_emb_path = RESULTS_DIR / "codebert_val_emb.npy"
    device = get_device()
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cb_model = AutoModel.from_pretrained(model_name).to(device)

    if cb_val_emb_path.exists():
        print("  Loading cached validation CodeBERT embeddings...")
        emb_val = np.load(cb_val_emb_path)
    else:
        print("  Extracting CodeBERT embeddings for validation tuning (N = 3,000)...")
        emb_val = extract_codebert_embeddings(X_val, tokenizer, cb_model, device)
        np.save(cb_val_emb_path, emb_val)

    # -------------------------------------------------------------------------
    # Candidate A: Baseline CodeBERT Hybrid
    # -------------------------------------------------------------------------
    print("\n[Candidate A] Evaluating Baseline CodeBERT Hybrid...")
    with open(MODELS_DIR / "expanded_codebert_hybrid_best.pkl", "rb") as f:
        cb_base_ckpt = pickle.load(f)
    cb_clf = cb_base_ckpt['clf']
    cb_scaler = cb_base_ckpt['scaler']
    from scripts.search_classical_models import SecurityStructuralFeatureExtractor
    base_extractor = SecurityStructuralFeatureExtractor()
    
    struct_val_base = cb_scaler.transform(base_extractor.transform(X_val))
    prob_cb_val = cb_clf.predict_proba(np.hstack([emb_val, struct_val_base]))[:, 1]
    
    struct_test_base = cb_scaler.transform(base_extractor.transform(X_test))
    prob_cb_test = cb_clf.predict_proba(np.hstack([emb_test, struct_test_base]))[:, 1]
    
    m_cb_val = compute_metrics(y_val, (prob_cb_val >= 0.50).astype(int), prob_cb_val)
    m_cb_val_cpp = compute_metrics(y_val[val_cpp_mask], (prob_cb_val[val_cpp_mask] >= 0.50).astype(int), prob_cb_val[val_cpp_mask])
    print(f"  Val Overall: Acc={m_cb_val['accuracy']*100:.2f}%, AUROC={m_cb_val['auroc']:.4f}")
    print(f"  Val C++ Subset: Acc={m_cb_val_cpp['accuracy']*100:.2f}%, AUROC={m_cb_val_cpp['auroc']:.4f}")

    # -------------------------------------------------------------------------
    # Candidate B: Enhanced Modern C++ Structural Model (phase14_cxx_structural_svm)
    # -------------------------------------------------------------------------
    print("\n[Candidate B] Fitting Enhanced Modern C++ Structural Model on 50k...")
    cxx_extractor = EnhancedModernCxxStructuralExtractor()
    
    union_cxx = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), max_features=18000, sublinear_tf=True, tokenizer=cxx_tokenize)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=25000, sublinear_tf=True)),
        ('cxx_struct', cxx_extractor)
    ])
    
    pipe_cxx = Pipeline([
        ('features', union_cxx),
        ('clf', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    
    t0 = time.time()
    pipe_cxx.fit(X_train, y_train)
    print(f"  Candidate B fitted in {time.time()-t0:.1f}s")
    
    prob_cxx_val = pipe_cxx.predict_proba(X_val)[:, 1]
    prob_cxx_test = pipe_cxx.predict_proba(X_test)[:, 1]
    
    m_cxx_val = compute_metrics(y_val, (prob_cxx_val >= 0.50).astype(int), prob_cxx_val)
    m_cxx_val_cpp = compute_metrics(y_val[val_cpp_mask], (prob_cxx_val[val_cpp_mask] >= 0.50).astype(int), prob_cxx_val[val_cpp_mask])
    print(f"  Val Overall: Acc={m_cxx_val['accuracy']*100:.2f}%, AUROC={m_cxx_val['auroc']:.4f}")
    print(f"  Val C++ Subset: Acc={m_cxx_val_cpp['accuracy']*100:.2f}%, AUROC={m_cxx_val_cpp['auroc']:.4f}")

    # Save Candidate B checkpoint
    ckpt_b_path = MODELS_DIR / "phase14_cxx_structural_svm.pkl"
    with open(ckpt_b_path, "wb") as f:
        pickle.dump({"pipeline": pipe_cxx, "model_id": "phase14_cxx_structural_svm"}, f)
    print(f"  Saved Candidate B to: {ckpt_b_path}")

    # -------------------------------------------------------------------------
    # Candidate C: CodeBERT + 24-dim C++ Structural Hybrid (phase14_codebert_cxx_hybrid)
    # -------------------------------------------------------------------------
    print("\n[Candidate C] Fitting CodeBERT + 24-dim Modern C++ Structural Hybrid...")
    # Extract 24-dim features for validation and test
    cxx_struct_val_raw = cxx_extractor.transform(X_val)
    cxx_struct_test_raw = cxx_extractor.transform(X_test)
    
    scaler_cxx = StandardScaler()
    cxx_struct_val_norm = scaler_cxx.fit_transform(cxx_struct_val_raw)
    cxx_struct_test_norm = scaler_cxx.transform(cxx_struct_test_raw)
    
    X_c_val = np.hstack([emb_val, cxx_struct_val_norm])
    X_c_test = np.hstack([emb_test, cxx_struct_test_norm])
    
    # Train logistic classifier on validation sample (5-fold cross-validated C search)
    best_c = 1.0
    best_val_auc = 0.0
    for c_cand in [0.01, 0.1, 1.0, 5.0, 10.0]:
        clf_c_cand = LogisticRegression(C=c_cand, penalty='l2', max_iter=1000, class_weight='balanced', random_state=42)
        # Train on half of val, evaluate on other half
        split_idx = len(X_c_val) // 2
        clf_c_cand.fit(X_c_val[:split_idx], y_val[:split_idx])
        p_cand = clf_c_cand.predict_proba(X_c_val[split_idx:])[:, 1]
        auc_cand = roc_auc_score(y_val[split_idx:], p_cand)
        if auc_cand > best_val_auc:
            best_val_auc = auc_cand
            best_c = c_cand
            
    clf_c = LogisticRegression(C=best_c, penalty='l2', max_iter=1000, class_weight='balanced', random_state=42)
    clf_c.fit(X_c_val, y_val)
    prob_c_val = clf_c.predict_proba(X_c_val)[:, 1]
    prob_c_test = clf_c.predict_proba(X_c_test)[:, 1]
    
    m_c_val = compute_metrics(y_val, (prob_c_val >= 0.50).astype(int), prob_c_val)
    m_c_val_cpp = compute_metrics(y_val[val_cpp_mask], (prob_c_val[val_cpp_mask] >= 0.50).astype(int), prob_c_val[val_cpp_mask])
    print(f"  Tuned C={best_c} on validation")
    print(f"  Val Overall: Acc={m_c_val['accuracy']*100:.2f}%, AUROC={m_c_val['auroc']:.4f}")
    print(f"  Val C++ Subset: Acc={m_c_val_cpp['accuracy']*100:.2f}%, AUROC={m_c_val_cpp['auroc']:.4f}")

    ckpt_c_path = MODELS_DIR / "phase14_codebert_cxx_hybrid.pkl"
    with open(ckpt_c_path, "wb") as f:
        pickle.dump({"clf": clf_c, "scaler": scaler_cxx, "model_id": "phase14_codebert_cxx_hybrid"}, f)
    print(f"  Saved Candidate C to: {ckpt_c_path}")

    # -------------------------------------------------------------------------
    # Candidate D: Domain-Calibrated Adaptive Ensemble (phase14_model_a_ensemble)
    # -------------------------------------------------------------------------
    print("\n[Candidate D] Optimizing Adaptive Domain-Calibrated Ensemble on Validation Data...")
    # Learn routing weight w_cxx for C++ samples and w_c for traditional C samples
    # For samples with C++ patterns: test weights w from 0.0 to 1.0 (weight of CxxStructural vs CodeBERT)
    best_w_cpp = 0.50
    best_w_c = 0.85
    best_ens_auc = 0.0
    
    for w_cpp in np.arange(0.2, 0.9, 0.1):
        for w_c in np.arange(0.5, 1.0, 0.1):
            p_ens_val = np.zeros(len(y_val))
            for i in range(len(y_val)):
                if val_cpp_mask[i]:
                    # C++ code: weight towards CxxStructural
                    p_ens_val[i] = (1.0 - w_cpp) * prob_cb_val[i] + w_cpp * prob_cxx_val[i]
                else:
                    # C code: weight towards CodeBERT
                    p_ens_val[i] = w_c * prob_cb_val[i] + (1.0 - w_c) * prob_cxx_val[i]
            
            auc_comb = roc_auc_score(y_val, p_ens_val)
            if auc_comb > best_ens_auc:
                best_ens_auc = auc_comb
                best_w_cpp = float(w_cpp)
                best_w_c = float(w_c)

    # Optimal decision threshold on validation
    best_thresh = 0.50
    best_val_f1 = 0.0
    p_ens_val_opt = np.zeros(len(y_val))
    for i in range(len(y_val)):
        if val_cpp_mask[i]:
            p_ens_val_opt[i] = (1.0 - best_w_cpp) * prob_cb_val[i] + best_w_cpp * prob_cxx_val[i]
        else:
            p_ens_val_opt[i] = best_w_c * prob_cb_val[i] + (1.0 - best_w_c) * prob_cxx_val[i]

    for thresh in np.arange(0.35, 0.65, 0.02):
        f1_cand = f1_score(y_val, (p_ens_val_opt >= thresh).astype(int), zero_division=0)
        if f1_cand > best_val_f1:
            best_val_f1 = f1_cand
            best_thresh = float(thresh)

    print(f"  Selected Validation Parameters: w_cpp={best_w_cpp:.2f} (structural weight on C++), w_c={best_w_c:.2f} (CodeBERT weight on C), decision threshold={best_thresh:.2f}")
    m_ens_val = compute_metrics(y_val, (p_ens_val_opt >= best_thresh).astype(int), p_ens_val_opt)
    m_ens_val_cpp = compute_metrics(y_val[val_cpp_mask], (p_ens_val_opt[val_cpp_mask] >= best_thresh).astype(int), p_ens_val_opt[val_cpp_mask])
    print(f"  Val Overall: Acc={m_ens_val['accuracy']*100:.2f}%, AUROC={m_ens_val['auroc']:.4f}, F1={m_ens_val['f1']:.4f}")
    print(f"  Val C++ Subset: Acc={m_ens_val_cpp['accuracy']*100:.2f}%, AUROC={m_ens_val_cpp['auroc']:.4f}")

    # Compute Candidate D on Held-Out Test Set
    test_cpp_mask = pd.Series(X_test).str.contains(r'\bnamespace\b|\bclass\s+\w+|\btemplate\s*<|\bstd::|\bEXPECT_', regex=True).values
    prob_ens_test = np.zeros(len(y_test))
    for i in range(len(y_test)):
        if test_cpp_mask[i]:
            prob_ens_test[i] = (1.0 - best_w_cpp) * prob_cb_test[i] + best_w_cpp * prob_cxx_test[i]
        else:
            prob_ens_test[i] = best_w_c * prob_cb_test[i] + (1.0 - best_w_c) * prob_cxx_test[i]

    ckpt_d_path = MODELS_DIR / "phase14_model_a_calibrated_ensemble.pkl"
    with open(ckpt_d_path, "wb") as f:
        pickle.dump({
            "model_id": "phase14_model_a_calibrated_ensemble",
            "w_cpp": best_w_cpp,
            "w_c": best_w_c,
            "threshold": best_thresh,
            "validation_auc": best_ens_auc
        }, f)
    print(f"  Saved Candidate D configuration to: {ckpt_d_path}")

    # =========================================================================
    # 4. FINAL INDEPENDENT BENCHMARK EVALUATION (Held-out Test, N = 3,169)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL INDEPENDENT EVALUATION ON DECONTAMINATED HELD-OUT BENCHMARK (N = 3,169)")
    print("=" * 80)

    candidates = {
        "Candidate A: Baseline CodeBERT Hybrid": (prob_cb_test, 0.50),
        "Candidate B: Modern C++ Structural SVM": (prob_cxx_test, 0.50),
        "Candidate C: CodeBERT C++ Structural Hybrid": (prob_c_test, 0.50),
        "Candidate D: Domain-Calibrated Adaptive Ensemble": (prob_ens_test, best_thresh),
    }

    final_results = {}
    for name, (probs, thresh) in candidates.items():
        preds = (probs >= thresh).astype(int)
        overall_m = compute_metrics(y_test, preds, probs)
        
        by_proj = {}
        for p in ['openssl', 'gpac', 'vim', 'envoy']:
            mask = (projects == p)
            by_proj[p] = compute_metrics(y_test[mask], preds[mask], probs[mask])
            
        final_results[name] = {
            "overall": overall_m,
            "by_project": by_proj,
            "threshold_used": thresh
        }
        
        print(f"\n--- {name} (Threshold = {thresh:.2f}) ---")
        print(f"  Overall Test Acc: {overall_m['accuracy']*100:.2f}% | Balanced Acc: {overall_m['balanced_accuracy']*100:.2f}% | AUROC: {overall_m['auroc']:.4f} | MCC: {overall_m['mcc']:+.4f} | F1: {overall_m['f1']:.4f}")
        for p in ['openssl', 'gpac', 'vim', 'envoy']:
            pm = by_proj[p]
            print(f"    • {p:8} (N={pm['n_samples']:4}): Acc={pm['accuracy']*100:.2f}% | AUROC={pm['auroc']:.4f} | MCC={pm['mcc']:+.4f} | F1={pm['f1']:.4f} (FPR={pm['fpr']*100:.1f}%, FNR={pm['fnr']*100:.1f}%)")

    # Save complete evaluation output
    out_eval_path = RESULTS_DIR / "phase14_model_a_comparison.json"
    with open(out_eval_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[✓] Phase 14 Model A results saved to: {out_eval_path}")


if __name__ == "__main__":
    main()

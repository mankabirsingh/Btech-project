#!/usr/bin/env python3
"""
Phase 16 — Experiment 5: Classical & Neural Stacking Ensemble
============================================================
Investigates an architectural meta-learner (stacking ensemble) that blends:
1. Classical N-gram & TF-IDF lexical models (expanded_tri_union_best) - high keyword precision
2. Deep Multimodal CodeBERT Hybrid (Phase 16 Omission-Aware Hybrid) - semantic context

Meta-Learner Protocol:
- Generates out-of-fold probability predictions on Validation Split (N = 3,000)
- Fits meta-learner (regularized logistic regression or calibrated blender) on Validation probabilities ONLY
- Evaluates zero-leakage generalisation on Clean Held-Out Benchmark (N = 3,169)
- Checkpoint saved to models/checkpoints/phase16_model_a_stacking_ensemble.pkl
"""

import sys
import os
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_P16_DIR = PROJECT_ROOT / "results" / "phase16"
RESULTS_P15_DIR = PROJECT_ROOT / "results" / "phase15"
RESULTS_P14_DIR = PROJECT_ROOT / "results" / "phase14"
RESULTS_P16_DIR.mkdir(parents=True, exist_ok=True)

from src.data.preprocessing import clean_code
from scripts.phase16_omission_extractor import Phase16OmissionAwareCxxExtractor


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
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
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
    print("PHASE 16 — EXPERIMENT 5: DUAL CLASSICAL-CODEBERT STACKING ENSEMBLE")
    print("=" * 80)

    # 1. Load Data
    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet")
    val_3k = val_df.sample(n=3000, random_state=42).reset_index(drop=True)
    X_val = val_3k['code'].values
    y_val = val_3k['vuln_label'].values.astype(int)

    test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet")
    X_test = test_df['code'].values
    y_test = test_df['vuln_label'].values.astype(int)
    projects = test_df['project'].values

    emb_val = np.load(RESULTS_P14_DIR / "codebert_val_emb.npy")
    emb_test = np.load(RESULTS_P14_DIR / "codebert_test_clean_emb.npy")

    # 2. Base Model 1: Classical Tri-Union Model (expanded_tri_union_best.pkl)
    print("\n[1] Running Base Model 1 (Classical Tri-Union)...")
    with open(MODELS_DIR / "expanded_tri_union_best.pkl", "rb") as f:
        tri_union_obj = pickle.load(f)
    pipe_tri = tri_union_obj["pipeline"]

    cleaned_val = [clean_code(c) for c in X_val]
    cleaned_test = [clean_code(c) for c in X_test]

    prob_tri_val = pipe_tri.predict_proba(cleaned_val)[:, 1]
    prob_tri_test = pipe_tri.predict_proba(cleaned_test)[:, 1]

    m_tri_test = compute_metrics(y_test, (prob_tri_test >= 0.50).astype(int), prob_tri_test)
    print(f"  Tri-Union Alone: Acc = {m_tri_test['accuracy']*100:.2f}%, AUROC = {m_tri_test['auroc']:.4f}, F1 = {m_tri_test['f1']:.4f}")

    # 3. Base Model 2: Phase 16 Omission-Aware CodeBERT Hybrid
    print("\n[2] Running Base Model 2 (Phase 16 CodeBERT Hybrid)...")
    cand1_path = MODELS_DIR / "phase16_codebert_omission_hybrid.pkl"
    target_ckpt = cand1_path if cand1_path.exists() else MODELS_DIR / "phase15_codebert_cxx_hybrid_best.pkl"
    with open(target_ckpt, "rb") as f:
        cb_ckpt = pickle.load(f)

    ext = Phase16OmissionAwareCxxExtractor() if len(cb_ckpt["feature_names"]) == 38 else None
    s_val = cb_ckpt["scaler"].transform(ext.transform(X_val)) if ext else np.zeros((len(X_val), 1))
    s_test = cb_ckpt["scaler"].transform(ext.transform(X_test)) if ext else np.zeros((len(X_test), 1))

    X_val_cb = np.hstack([emb_val, s_val])
    X_test_cb = np.hstack([emb_test, s_test])

    prob_cb_val = cb_ckpt["clf"].predict_proba(X_val_cb)[:, 1]
    prob_cb_test = cb_ckpt["clf"].predict_proba(X_test_cb)[:, 1]

    m_cb_test = compute_metrics(y_test, (prob_cb_test >= 0.50).astype(int), prob_cb_test)
    print(f"  CodeBERT Hybrid Alone: Acc = {m_cb_test['accuracy']*100:.2f}%, AUROC = {m_cb_test['auroc']:.4f}, F1 = {m_cb_test['f1']:.4f}")

    # 4. Meta-Learner Fitting on Validation Split ONLY
    print("\n[3] Fitting Stacking Meta-Learner on Validation Split...")
    eps = 1e-4
    logit_tri_val = np.log((prob_tri_val + eps) / (1.0 - prob_tri_val + eps))
    logit_cb_val = np.log((prob_cb_val + eps) / (1.0 - prob_cb_val + eps))

    logit_tri_test = np.log((prob_tri_test + eps) / (1.0 - prob_tri_test + eps))
    logit_cb_test = np.log((prob_cb_test + eps) / (1.0 - prob_cb_test + eps))

    # Features: [logit_tri, logit_cb, diff, product]
    Z_val = np.column_stack([logit_tri_val, logit_cb_val, logit_cb_val - logit_tri_val])
    Z_test = np.column_stack([logit_tri_test, logit_cb_test, logit_cb_test - logit_tri_test])

    meta_clf = LogisticRegression(C=0.5, penalty='l2', random_state=42)
    meta_clf.fit(Z_val, y_val)

    prob_meta_val = meta_clf.predict_proba(Z_val)[:, 1]
    prob_meta_test = meta_clf.predict_proba(Z_test)[:, 1]

    # Select threshold on validation
    best_tau = 0.50
    best_f1 = 0.0
    for tau in np.linspace(0.30, 0.70, 41):
        f1_v = f1_score(y_val, (prob_meta_val >= tau).astype(int), zero_division=0)
        if f1_v > best_f1:
            best_f1 = f1_v
            best_tau = float(tau)

    print(f"  Meta-Learner Weights: Tri-Union = {meta_clf.coef_[0][0]:.3f}, CodeBERT = {meta_clf.coef_[0][1]:.3f}, Diff = {meta_clf.coef_[0][2]:.3f}")
    print(f"  Validation Optimal Tau: {best_tau:.3f} (Val F1 = {best_f1:.4f})")

    # 5. Evaluate on Held-Out Test Set
    pred_meta_test = (prob_meta_test >= best_tau).astype(int)
    m_meta_test = compute_metrics(y_test, pred_meta_test, prob_meta_test)

    print(f"\nStacking Ensemble Performance on Held-Out Benchmark (N = 3,169):")
    print(f"  Accuracy:          {m_meta_test['accuracy']*100:.2f}%")
    print(f"  AUROC:             {m_meta_test['auroc']:.4f}")
    print(f"  F1 Score:          {m_meta_test['f1']:.4f}")
    print(f"  Precision:         {m_meta_test['precision']*100:.2f}%")
    print(f"  Recall:            {m_meta_test['recall']*100:.2f}%")
    print(f"  FPR:               {m_meta_test['fpr']*100:.2f}%")
    print(f"  Brier Score:       {m_meta_test['brier_score']:.4f}")

    # Project-wise breakdown
    print(f"\nProject-wise Breakdown for Stacking Ensemble:")
    proj_meta = {}
    for p in sorted(np.unique(projects)):
        p_mask = (projects == p)
        m_p = compute_metrics(y_test[p_mask], pred_meta_test[p_mask], prob_meta_test[p_mask])
        proj_meta[p] = m_p
        print(f"  • {p:10}: Acc = {m_p['accuracy']*100:.2f}%, AUROC = {m_p['auroc']:.4f}, FPR = {m_p['fpr']*100:.1f}%, Rec = {m_p['recall']*100:.1f}%")

    # Save Checkpoint
    ckpt_meta_path = MODELS_DIR / "phase16_model_a_stacking_ensemble.pkl"
    with open(ckpt_meta_path, "wb") as f:
        pickle.dump({
            "meta_clf": meta_clf,
            "weights": meta_clf.coef_[0].tolist(),
            "intercept": float(meta_clf.intercept_[0]),
            "best_tau": best_tau,
            "model_id": "phase16_model_a_stacking_ensemble"
        }, f)
    print(f"\n  Saved Stacking Ensemble Checkpoint to: {ckpt_meta_path}")

    # Save Experiment 5 results
    out_exp5_path = RESULTS_P16_DIR / "exp5_stacking_ensemble.json"
    with open(out_exp5_path, "w") as f:
        json.dump({
            "tri_union_alone": m_tri_test,
            "codebert_alone": m_cb_test,
            "stacking_ensemble": m_meta_test,
            "project_breakdown": proj_meta,
            "meta_weights": meta_clf.coef_[0].tolist(),
            "best_tau": best_tau
        }, f, indent=2)

    print(f"[✓] Experiment 5 results saved to: {out_exp5_path}")


if __name__ == "__main__":
    main()

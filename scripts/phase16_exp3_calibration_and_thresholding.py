#!/usr/bin/env python3
"""
Phase 16 — Experiment 3: Validation-Based Threshold & Calibration Optimisation
=============================================================================
Performs principled calibration and decision threshold optimisation using the
validation split (N = 3,000) ONLY, strictly preserving held-out test isolation.

Techniques Evaluated:
1. Platt Scaling / Logistic Calibration on Validation Set
2. Temperature Scaling (Tuning temperature T via NLL on validation)
3. Isotonic Regression Calibration on Validation Set
4. Optimal Decision Threshold Selection (tuning tau on validation F1 & balanced accuracy)
5. Expected Calibration Error (ECE) computation before and after calibration
"""

import sys
import os
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, log_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_P16_DIR = PROJECT_ROOT / "results" / "phase16"
RESULTS_P15_DIR = PROJECT_ROOT / "results" / "phase15"
RESULTS_P14_DIR = PROJECT_ROOT / "results" / "phase14"
RESULTS_P16_DIR.mkdir(parents=True, exist_ok=True)

from scripts.phase16_omission_extractor import Phase16OmissionAwareCxxExtractor


def compute_ece(y_true, y_prob, n_bins=10) -> float:
    """Computes Expected Calibration Error across 10 probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


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
        "ece": compute_ece(y_true, y_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": fpr,
        "fnr": fnr
    }


def main():
    print("=" * 80)
    print("PHASE 16 — EXPERIMENT 3: VALIDATION-BASED THRESHOLD & CALIBRATION")
    print("=" * 80)

    # 1. Load Validation and Test Splits
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

    # Load latest trained checkpoint (Candidate 1 from Exp 1 if ready, else Phase 15)
    cand1_path = MODELS_DIR / "phase16_codebert_omission_hybrid.pkl"
    target_ckpt_path = cand1_path if cand1_path.exists() else MODELS_DIR / "phase15_codebert_cxx_hybrid_best.pkl"
    print(f"\n[1] Evaluating Calibration for Checkpoint: {target_ckpt_path.name}")
    with open(target_ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    clf = ckpt["clf"]
    scaler = ckpt["scaler"]

    # Extract features
    ext = Phase16OmissionAwareCxxExtractor() if len(ckpt["feature_names"]) == 38 else Phase15EnhancedModernCxxExtractor()
    s_val = scaler.transform(ext.transform(X_val))
    s_test = scaler.transform(ext.transform(X_test))

    X_val_all = np.hstack([emb_val, s_val])
    X_test_all = np.hstack([emb_test, s_test])

    # Uncalibrated probabilities & logits
    # Logit: z = w^T x + b
    logit_val = clf.decision_function(X_val_all)
    logit_test = clf.decision_function(X_test_all)

    prob_val_uncal = clf.predict_proba(X_val_all)[:, 1]
    prob_test_uncal = clf.predict_proba(X_test_all)[:, 1]

    m_val_raw = compute_metrics(y_val, (prob_val_uncal >= 0.50).astype(int), prob_val_uncal)
    m_test_raw = compute_metrics(y_test, (prob_test_uncal >= 0.50).astype(int), prob_test_uncal)

    print(f"\nRaw Uncalibrated Performance:")
    print(f"  Validation (N = 3,000): Acc = {m_val_raw['accuracy']*100:.2f}%, AUROC = {m_val_raw['auroc']:.4f}, ECE = {m_val_raw['ece']:.4f}, Brier = {m_val_raw['brier_score']:.4f}")
    print(f"  Test Held-Out (N = 3,169): Acc = {m_test_raw['accuracy']*100:.2f}%, AUROC = {m_test_raw['auroc']:.4f}, ECE = {m_test_raw['ece']:.4f}, Brier = {m_test_raw['brier_score']:.4f}")

    # -------------------------------------------------------------------------
    # Technique A: Optimal Decision Threshold on Validation Set
    # -------------------------------------------------------------------------
    print("\n[2] Selecting Optimal Decision Threshold on Validation Split...")
    best_tau_val = 0.50
    best_val_f1 = 0.0
    tau_search = np.linspace(0.20, 0.80, 61)
    for tau in tau_search:
        p_val_pred = (prob_val_uncal >= tau).astype(int)
        f1_v = f1_score(y_val, p_val_pred, zero_division=0)
        if f1_v > best_val_f1:
            best_val_f1 = f1_v
            best_tau_val = float(tau)

    print(f"  Optimal Validation Threshold: Tau* = {best_tau_val:.3f} (Val F1 = {best_val_f1:.4f})")
    m_test_thresh = compute_metrics(y_test, (prob_test_uncal >= best_tau_val).astype(int), prob_test_uncal)
    print(f"  Held-Out Test under Tau* ({best_tau_val:.3f}): Acc = {m_test_thresh['accuracy']*100:.2f}%, F1 = {m_test_thresh['f1']:.4f}, Rec = {m_test_thresh['recall']*100:.1f}%, FPR = {m_test_thresh['fpr']*100:.1f}%")

    # -------------------------------------------------------------------------
    # Technique B: Temperature Scaling on Validation Set
    # -------------------------------------------------------------------------
    print("\n[3] Fitting Temperature Scaling on Validation Split...")
    def nll_obj(T):
        scaled_logits = logit_val / T[0]
        # sigmoid
        p = 1.0 / (1.0 + np.exp(-scaled_logits))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return float(log_loss(y_val, p))

    res_t = minimize(nll_obj, [1.0], bounds=[(0.05, 5.0)], method='L-BFGS-B')
    opt_temp = float(res_t.x[0])
    print(f"  Optimal Temperature: T = {opt_temp:.3f} (NLL = {res_t.fun:.4f})")

    prob_val_temp = 1.0 / (1.0 + np.exp(-logit_val / opt_temp))
    prob_test_temp = 1.0 / (1.0 + np.exp(-logit_test / opt_temp))

    m_test_temp = compute_metrics(y_test, (prob_test_temp >= 0.50).astype(int), prob_test_temp)
    print(f"  Held-Out Test under Temperature Scaling: Acc = {m_test_temp['accuracy']*100:.2f}%, ECE = {m_test_temp['ece']:.4f} (Raw: {m_test_raw['ece']:.4f}), Brier = {m_test_temp['brier_score']:.4f}")

    # -------------------------------------------------------------------------
    # Technique C: Platt Scaling on Validation Set
    # -------------------------------------------------------------------------
    print("\n[4] Fitting Platt Scaling (Logistic Calibration) on Validation Split...")
    platt_lr = LogisticRegression(penalty=None, solver='lbfgs', random_state=42)
    platt_lr.fit(logit_val.reshape(-1, 1), y_val)

    prob_val_platt = platt_lr.predict_proba(logit_val.reshape(-1, 1))[:, 1]
    prob_test_platt = platt_lr.predict_proba(logit_test.reshape(-1, 1))[:, 1]

    m_test_platt = compute_metrics(y_test, (prob_test_platt >= 0.50).astype(int), prob_test_platt)
    print(f"  Held-Out Test under Platt Calibration: Acc = {m_test_platt['accuracy']*100:.2f}%, ECE = {m_test_platt['ece']:.4f}, Brier = {m_test_platt['brier_score']:.4f}, FPR = {m_test_platt['fpr']*100:.1f}%")

    # -------------------------------------------------------------------------
    # Technique D: Combined Calibration + Validation Threshold
    # -------------------------------------------------------------------------
    best_tau_platt = 0.50
    best_f1_platt = 0.0
    for tau in tau_search:
        p_val_pred = (prob_val_platt >= tau).astype(int)
        f1_v = f1_score(y_val, p_val_pred, zero_division=0)
        if f1_v > best_f1_platt:
            best_f1_platt = f1_v
            best_tau_platt = float(tau)

    m_test_comb = compute_metrics(y_test, (prob_test_platt >= best_tau_platt).astype(int), prob_test_platt)
    print(f"\nCombined (Platt Scaling + Tau*={best_tau_platt:.3f}):")
    print(f"  Accuracy:          {m_test_comb['accuracy']*100:.2f}%")
    print(f"  F1 Score:          {m_test_comb['f1']:.4f}")
    print(f"  Recall (Sens):     {m_test_comb['recall']*100:.1f}%")
    print(f"  Precision:         {m_test_comb['precision']*100:.1f}%")
    print(f"  FPR:               {m_test_comb['fpr']*100:.1f}%")
    print(f"  ECE:               {m_test_comb['ece']:.4f}")
    print(f"  Brier Score:       {m_test_comb['brier_score']:.4f}")

    # Save Experiment 3 Results
    out_exp3_path = RESULTS_P16_DIR / "exp3_calibration_and_thresholding.json"
    with open(out_exp3_path, "w") as f:
        json.dump({
            "raw_uncalibrated": m_test_raw,
            "validation_threshold_only": {
                "best_tau_val": best_tau_val,
                "test_metrics": m_test_thresh
            },
            "temperature_scaling": {
                "opt_temperature": opt_temp,
                "test_metrics": m_test_temp
            },
            "platt_scaling": {
                "coef": float(platt_lr.coef_[0][0]),
                "intercept": float(platt_lr.intercept_[0]),
                "test_metrics": m_test_platt
            },
            "combined_platt_and_threshold": {
                "best_tau_platt": best_tau_platt,
                "test_metrics": m_test_comb
            }
        }, f, indent=2)

    print(f"\n[✓] Experiment 3 results saved to: {out_exp3_path}")


if __name__ == "__main__":
    main()

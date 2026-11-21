#!/usr/bin/env python3
"""
Phase 15 — Step 1: Baseline Reproduction
========================================
Reproduces and logs the performance of the three existing Model A candidate checkpoints:
1. expanded_tri_union_best.pkl (Production Fast Baseline)
2. expanded_codebert_hybrid_best.pkl (Research Champion Baseline)
3. phase14_codebert_cxx_hybrid.pkl (Phase 14 Modern C++ Champion)

Evaluates on:
- Clean Held-Out Benchmark (N = 3,169: OpenSSL, GPAC, Vim, Envoy)
- Project-level breakdowns with complete confusion matrices and 95% Wilson CIs
- Envoy sub-cohort breakdown (GoogleTest Scaffolding N=281 vs Production Code N=322)
"""

import sys
import os
import json
import pickle
import re
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_P14_DIR = PROJECT_ROOT / "results" / "phase14"
RESULTS_P15_DIR = PROJECT_ROOT / "results" / "phase15"
RESULTS_P15_DIR.mkdir(parents=True, exist_ok=True)

from src.models.cxx_features import EnhancedModernCxxStructuralExtractor
from scripts.search_classical_models import SecurityStructuralFeatureExtractor


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    denom = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


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

    acc = float(accuracy_score(y_true, y_pred))
    ci_low, ci_high = wilson_score_interval(int(np.sum(y_true == y_pred)), len(y_true))

    return {
        "n_samples": int(len(y_true)),
        "accuracy": acc,
        "accuracy_ci_95": [float(ci_low), float(ci_high)],
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
    print("PHASE 15 — STEP 1: BASELINE REPRODUCTION")
    print("=" * 80)

    # 1. Load clean held-out benchmark
    clean_test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet")
    X_test = clean_test_df['code'].values
    y_test = clean_test_df['vuln_label'].values.astype(int)
    projects = clean_test_df['project'].values

    # Pre-extracted CodeBERT test embeddings
    emb_test = np.load(RESULTS_P14_DIR / "codebert_test_clean_emb.npy")
    print(f"Loaded held-out test data: {len(X_test)} samples ({y_test.mean()*100:.1f}% vulnerable)")
    print(f"Loaded CodeBERT test embeddings: {emb_test.shape}")

    # Detect GoogleTest on Envoy
    gtest_regex = re.compile(r'\b(EXPECT_CALL|ON_CALL|TEST_F|TEST_P|TEST|MOCK_METHOD|ASSERT_TRUE|ASSERT_FALSE|EXPECT_TRUE|EXPECT_FALSE|EXPECT_EQ|ASSERT_EQ)\b')
    is_gtest = np.array([bool(gtest_regex.search(c)) for c in X_test])

    results = {}

    # -------------------------------------------------------------------------
    # Baseline 1: expanded_tri_union_best.pkl
    # -------------------------------------------------------------------------
    print("\n[1] Evaluating expanded_tri_union_best.pkl...")
    with open(MODELS_DIR / "expanded_tri_union_best.pkl", "rb") as f:
        tri_ckpt = pickle.load(f)
    tri_pipe = tri_ckpt['pipeline'] if 'pipeline' in tri_ckpt else tri_ckpt
    
    # Predict probabilities (modified huber or decision function)
    if hasattr(tri_pipe, "predict_proba"):
        prob_tri = tri_pipe.predict_proba(X_test)[:, 1]
    else:
        scores = tri_pipe.decision_function(X_test)
        prob_tri = 1.0 / (1.0 + np.exp(-scores))
    pred_tri = (prob_tri >= 0.50).astype(int)

    m_tri_overall = compute_metrics(y_test, pred_tri, prob_tri)
    m_tri_proj = {}
    for p in ['openssl', 'gpac', 'vim', 'envoy']:
        mask = (projects == p)
        m_tri_proj[p] = compute_metrics(y_test[mask], pred_tri[mask], prob_tri[mask])
    envoy_mask = (projects == 'envoy')
    m_tri_envoy_gtest = compute_metrics(y_test[envoy_mask & is_gtest], pred_tri[envoy_mask & is_gtest], prob_tri[envoy_mask & is_gtest])
    m_tri_envoy_prod = compute_metrics(y_test[envoy_mask & (~is_gtest)], pred_tri[envoy_mask & (~is_gtest)], prob_tri[envoy_mask & (~is_gtest)])

    results["expanded_tri_union_best"] = {
        "overall": m_tri_overall,
        "by_project": m_tri_proj,
        "envoy_subcohorts": {"gtest": m_tri_envoy_gtest, "production": m_tri_envoy_prod}
    }
    print(f"  Overall: Acc={m_tri_overall['accuracy']*100:.2f}% | AUROC={m_tri_overall['auroc']:.4f} | F1={m_tri_overall['f1']:.4f}")

    # -------------------------------------------------------------------------
    # Baseline 2: expanded_codebert_hybrid_best.pkl
    # -------------------------------------------------------------------------
    print("\n[2] Evaluating expanded_codebert_hybrid_best.pkl...")
    with open(MODELS_DIR / "expanded_codebert_hybrid_best.pkl", "rb") as f:
        cb_ckpt = pickle.load(f)
    cb_clf = cb_ckpt['clf']
    cb_scaler = cb_ckpt['scaler']
    extractor_base = SecurityStructuralFeatureExtractor()
    struct_base = cb_scaler.transform(extractor_base.transform(X_test))
    X_cb_test = np.hstack([emb_test, struct_base])
    prob_cb = cb_clf.predict_proba(X_cb_test)[:, 1]
    pred_cb = (prob_cb >= 0.50).astype(int)

    m_cb_overall = compute_metrics(y_test, pred_cb, prob_cb)
    m_cb_proj = {}
    for p in ['openssl', 'gpac', 'vim', 'envoy']:
        mask = (projects == p)
        m_cb_proj[p] = compute_metrics(y_test[mask], pred_cb[mask], prob_cb[mask])
    m_cb_envoy_gtest = compute_metrics(y_test[envoy_mask & is_gtest], pred_cb[envoy_mask & is_gtest], prob_cb[envoy_mask & is_gtest])
    m_cb_envoy_prod = compute_metrics(y_test[envoy_mask & (~is_gtest)], pred_cb[envoy_mask & (~is_gtest)], prob_cb[envoy_mask & (~is_gtest)])

    results["expanded_codebert_hybrid_best"] = {
        "overall": m_cb_overall,
        "by_project": m_cb_proj,
        "envoy_subcohorts": {"gtest": m_cb_envoy_gtest, "production": m_cb_envoy_prod}
    }
    print(f"  Overall: Acc={m_cb_overall['accuracy']*100:.2f}% | AUROC={m_cb_overall['auroc']:.4f} | F1={m_cb_overall['f1']:.4f}")

    # -------------------------------------------------------------------------
    # Baseline 3: phase14_codebert_cxx_hybrid.pkl
    # -------------------------------------------------------------------------
    print("\n[3] Evaluating phase14_codebert_cxx_hybrid.pkl...")
    with open(MODELS_DIR / "phase14_codebert_cxx_hybrid.pkl", "rb") as f:
        cxx_ckpt = pickle.load(f)
    cxx_clf = cxx_ckpt['clf']
    cxx_scaler = cxx_ckpt['scaler']
    extractor_cxx = EnhancedModernCxxStructuralExtractor()
    struct_cxx = cxx_scaler.transform(extractor_cxx.transform(X_test))
    X_cxx_test = np.hstack([emb_test, struct_cxx])
    prob_cxx = cxx_clf.predict_proba(X_cxx_test)[:, 1]
    pred_cxx = (prob_cxx >= 0.50).astype(int)

    m_cxx_overall = compute_metrics(y_test, pred_cxx, prob_cxx)
    m_cxx_proj = {}
    for p in ['openssl', 'gpac', 'vim', 'envoy']:
        mask = (projects == p)
        m_cxx_proj[p] = compute_metrics(y_test[mask], pred_cxx[mask], prob_cxx[mask])
    m_cxx_envoy_gtest = compute_metrics(y_test[envoy_mask & is_gtest], pred_cxx[envoy_mask & is_gtest], prob_cxx[envoy_mask & is_gtest])
    m_cxx_envoy_prod = compute_metrics(y_test[envoy_mask & (~is_gtest)], pred_cxx[envoy_mask & (~is_gtest)], prob_cxx[envoy_mask & (~is_gtest)])

    results["phase14_codebert_cxx_hybrid"] = {
        "overall": m_cxx_overall,
        "by_project": m_cxx_proj,
        "envoy_subcohorts": {"gtest": m_cxx_envoy_gtest, "production": m_cxx_envoy_prod}
    }
    print(f"  Overall: Acc={m_cxx_overall['accuracy']*100:.2f}% | AUROC={m_cxx_overall['auroc']:.4f} | F1={m_cxx_overall['f1']:.4f}")

    # Save to JSON
    out_path = RESULTS_P15_DIR / "baseline_reproduction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Baseline reproduction successfully written to: {out_path}")


if __name__ == "__main__":
    main()

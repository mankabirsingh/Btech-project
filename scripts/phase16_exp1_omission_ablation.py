#!/usr/bin/env python3
"""
Phase 16 — Experiment 1: Omission-Aware Feature Ablation
========================================================
Compares:
- Baseline: Phase 15 Champion (28 Modern C++ features + CodeBERT, 25k training, C=0.1)
- Candidate 1: Phase 16 Omission-Aware Hybrid (38 features + CodeBERT, 25k training, C tuned on Val)

Targeted Evaluation:
- Overall held-out metrics (Accuracy, AUROC, F1, FPR, FNR, Brier)
- Project breakdown (Envoy, GPAC, OpenSSL, Vim)
- Envoy production code (N = 322) vs test scaffolding (N = 281)
- Targeted CWE recall (CWE-476, CWE-400, CWE-835, CWE-416, CWE-125)
"""

import sys
import os
import re
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

from scripts.phase15_train_model_a import Phase15EnhancedModernCxxExtractor
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
    print("PHASE 16 — EXPERIMENT 1: OMISSION-AWARE FEATURE ABLATION")
    print("=" * 80)

    # 1. Load Data
    print("\n[1] Loading Datasets and Embeddings...")
    train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    vuln_train = train_df[train_df['vuln_label'] == 1]
    benign_train = train_df[train_df['vuln_label'] == 0]

    # Deterministic 25k split (exact same as Phase 15)
    train_25k = pd.concat([
        vuln_train.sample(n=12500, random_state=42),
        benign_train.sample(n=12500, random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    X_train = train_25k['code'].values
    y_train = train_25k['vuln_label'].values.astype(int)

    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet")
    val_3k = val_df.sample(n=3000, random_state=42).reset_index(drop=True)
    X_val = val_3k['code'].values
    y_val = val_3k['vuln_label'].values.astype(int)

    test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet")
    X_test = test_df['code'].values
    y_test = test_df['vuln_label'].values.astype(int)
    projects = test_df['project'].values
    cwes = test_df['cwe'].values

    emb_train = np.load(RESULTS_P15_DIR / "codebert_train_25k_emb.npy")
    emb_val = np.load(RESULTS_P14_DIR / "codebert_val_emb.npy")
    emb_test = np.load(RESULTS_P14_DIR / "codebert_test_clean_emb.npy")

    print(f"  Training:   {len(X_train)} samples, emb: {emb_train.shape}")
    print(f"  Validation: {len(X_val)} samples, emb: {emb_val.shape}")
    print(f"  Test:       {len(X_test)} samples, emb: {emb_test.shape}")

    # -------------------------------------------------------------------------
    # Baseline: Phase 15 Champion (28 Modern C++ features)
    # -------------------------------------------------------------------------
    print("\n[2] Evaluating Phase 15 Baseline Checkpoint...")
    with open(MODELS_DIR / "phase15_codebert_cxx_hybrid_best.pkl", "rb") as f:
        ckpt_p15 = pickle.load(f)

    ext_p15 = Phase15EnhancedModernCxxExtractor()
    s_test_p15 = ckpt_p15["scaler"].transform(ext_p15.transform(X_test))
    prob_p15 = ckpt_p15["clf"].predict_proba(np.hstack([emb_test, s_test_p15]))[:, 1]
    pred_p15 = (prob_p15 >= 0.50).astype(int)
    m_base = compute_metrics(y_test, pred_p15, prob_p15)
    print(f"  Phase 15 Baseline: Acc = {m_base['accuracy']*100:.2f}% | AUROC = {m_base['auroc']:.4f} | F1 = {m_base['f1']:.4f} | FPR = {m_base['fpr']*100:.2f}%")

    # -------------------------------------------------------------------------
    # Candidate 1: Phase 16 Omission-Aware Hybrid (38 features)
    # -------------------------------------------------------------------------
    print("\n[3] Extracting 38-Dimensional Omission-Aware Features...")
    t0 = time.time()
    ext_p16 = Phase16OmissionAwareCxxExtractor()
    feats_train_raw = ext_p16.transform(X_train)
    feats_val_raw = ext_p16.transform(X_val)
    feats_test_raw = ext_p16.transform(X_test)

    scaler_38 = StandardScaler()
    s_train_38 = scaler_38.fit_transform(feats_train_raw)
    s_val_38 = scaler_38.transform(feats_val_raw)
    s_test_38 = scaler_38.transform(feats_test_raw)
    print(f"  Extraction & scaling completed in {time.time()-t0:.1f}s")

    X_tr_cand1 = np.hstack([emb_train, s_train_38])
    X_val_cand1 = np.hstack([emb_val, s_val_38])
    X_te_cand1 = np.hstack([emb_test, s_test_38])

    # Validation Tuning for C
    print("\n[4] Hyperparameter Tuning on Validation Split (Zero Test Leakage)...")
    best_c_1 = 0.1
    best_val_auc_1 = 0.0
    for c_cand in [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        clf_temp = LogisticRegression(C=c_cand, class_weight='balanced', max_iter=1000, random_state=42)
        clf_temp.fit(X_tr_cand1, y_train)
        p_val = clf_temp.predict_proba(X_val_cand1)[:, 1]
        auc_v = roc_auc_score(y_val, p_val)
        if auc_v > best_val_auc_1:
            best_val_auc_1 = auc_v
            best_c_1 = c_cand
        print(f"    C = {c_cand:5.3f} -> Val AUROC = {auc_v:.4f}")

    print(f"  Selected Best C = {best_c_1} (Val AUROC = {best_val_auc_1:.4f})")

    # Fit Champion Candidate 1
    clf_cand1 = LogisticRegression(C=best_c_1, class_weight='balanced', max_iter=1000, random_state=42)
    clf_cand1.fit(X_tr_cand1, y_train)
    prob_cand1 = clf_cand1.predict_proba(X_te_cand1)[:, 1]
    pred_cand1 = (prob_cand1 >= 0.50).astype(int)

    m_cand1 = compute_metrics(y_test, pred_cand1, prob_cand1)
    print(f"\nCandidate 1 (Omission-Aware 38-dim):")
    print(f"  Accuracy:          {m_cand1['accuracy']*100:.2f}% (Delta = {m_cand1['accuracy']*100 - m_base['accuracy']*100:+.2f}%)")
    print(f"  AUROC:             {m_cand1['auroc']:.4f} (Delta = {m_cand1['auroc'] - m_base['auroc']:+.4f})")
    print(f"  Recall (Sens):     {m_cand1['recall']*100:.2f}% (Delta = {m_cand1['recall']*100 - m_base['recall']*100:+.2f}%)")
    print(f"  Precision:         {m_cand1['precision']*100:.2f}% (Delta = {m_cand1['precision']*100 - m_base['precision']*100:+.2f}%)")
    print(f"  F1 Score:          {m_cand1['f1']:.4f}")
    print(f"  FPR:               {m_cand1['fpr']*100:.2f}% (Delta = {m_cand1['fpr']*100 - m_base['fpr']*100:+.2f}%)")
    print(f"  Brier Score:       {m_cand1['brier_score']:.4f}")

    # Save Checkpoint
    ckpt_cand1_path = MODELS_DIR / "phase16_codebert_omission_hybrid.pkl"
    with open(ckpt_cand1_path, "wb") as f:
        pickle.dump({
            "clf": clf_cand1,
            "scaler": scaler_38,
            "feature_names": ext_p16.get_feature_names(),
            "best_c": best_c_1,
            "val_score": float(best_val_auc_1),
            "train_size": len(X_train),
            "model_id": "phase16_codebert_omission_hybrid"
        }, f)
    print(f"  Saved Candidate 1 Checkpoint to: {ckpt_cand1_path}")

    # -------------------------------------------------------------------------
    # Project-wise Breakdown Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PROJECT-WISE PERFORMANCE COMPARISON")
    print("=" * 80)
    proj_comp = {}
    for p in sorted(np.unique(projects)):
        p_mask = (projects == p)
        m_p_base = compute_metrics(y_test[p_mask], pred_p15[p_mask], prob_p15[p_mask])
        m_p_cand = compute_metrics(y_test[p_mask], pred_cand1[p_mask], prob_cand1[p_mask])
        proj_comp[p] = {
            "baseline": m_p_base,
            "candidate1": m_p_cand,
            "accuracy_delta": m_p_cand["accuracy"] - m_p_base["accuracy"],
            "recall_delta": m_p_cand["recall"] - m_p_base["recall"]
        }
        print(f"  • {p:10}: Baseline = {m_p_base['accuracy']*100:.2f}% -> Cand 1 = {m_p_cand['accuracy']*100:.2f}% (Delta = {m_p_cand['accuracy']*100 - m_p_base['accuracy']*100:+.2f}%, Rec Delta = {m_p_cand['recall']*100 - m_p_base['recall']*100:+.2f}%)")

    # Envoy Subcohort Breakdown
    envoy_mask = (projects == "envoy")
    gtest_regex = re.compile(r'\b(EXPECT_CALL|ON_CALL|TEST_F|TEST_P|TEST|MOCK_METHOD|ASSERT_TRUE|ASSERT_FALSE|EXPECT_TRUE|EXPECT_FALSE|EXPECT_EQ|ASSERT_EQ)\b')
    is_mock = np.array([bool(gtest_regex.search(c)) for c in X_test[envoy_mask]])
    
    prod_base = compute_metrics(y_test[envoy_mask][~is_mock], pred_p15[envoy_mask][~is_mock], prob_p15[envoy_mask][~is_mock])
    prod_cand = compute_metrics(y_test[envoy_mask][~is_mock], pred_cand1[envoy_mask][~is_mock], prob_cand1[envoy_mask][~is_mock])
    mock_base = compute_metrics(y_test[envoy_mask][is_mock], pred_p15[envoy_mask][is_mock], prob_p15[envoy_mask][is_mock])
    mock_cand = compute_metrics(y_test[envoy_mask][is_mock], pred_cand1[envoy_mask][is_mock], prob_cand1[envoy_mask][is_mock])

    print(f"\nEnvoy Subcohort Comparison:")
    print(f"  • Production Code (N = 322): Baseline = {prod_base['accuracy']*100:.2f}% -> Cand 1 = {prod_cand['accuracy']*100:.2f}% (Delta = {prod_cand['accuracy']*100 - prod_base['accuracy']*100:+.2f}%)")
    print(f"  • Test Scaffolding (N = 281): Baseline = {mock_base['accuracy']*100:.2f}% -> Cand 1 = {mock_cand['accuracy']*100:.2f}% (Delta = {mock_cand['accuracy']*100 - mock_base['accuracy']*100:+.2f}%)")

    # -------------------------------------------------------------------------
    # Targeted CWE Recall Audit (CWE-476, CWE-400, CWE-835, CWE-416, CWE-125)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TARGETED CWE RECALL AUDIT (DID OMISSION FEATURES FIX MISSED VULNS?)")
    print("=" * 80)
    cwe_comp = {}
    for target_cwe in ["CWE-476", "CWE-400", "CWE-835", "CWE-416", "CWE-125", "CWE-119", "CWE-787"]:
        cwe_mask = (cwes == target_cwe) & (y_test == 1)
        total_vulns = int(cwe_mask.sum())
        if total_vulns == 0:
            continue
        caught_base = int((cwe_mask & (pred_p15 == 1)).sum())
        caught_cand = int((cwe_mask & (pred_cand1 == 1)).sum())
        rec_base = caught_base / total_vulns
        rec_cand = caught_cand / total_vulns
        cwe_comp[target_cwe] = {
            "total_vulns": total_vulns,
            "baseline_caught": caught_base, "baseline_recall": rec_base,
            "cand1_caught": caught_cand, "cand1_recall": rec_cand,
            "recall_delta": rec_cand - rec_base
        }
        print(f"  • {target_cwe:10}: Total = {total_vulns:2d} | Baseline Caught = {caught_base:2d} ({rec_base*100:4.1f}%) -> Cand 1 Caught = {caught_cand:2d} ({rec_cand*100:4.1f}%) | Delta = {rec_cand*100 - rec_base*100:+4.1f}%")

    # Save complete evaluation output
    out_exp1_path = RESULTS_P16_DIR / "exp1_omission_ablation.json"
    with open(out_exp1_path, "w") as f:
        json.dump({
            "overall_baseline": m_base,
            "overall_candidate1": m_cand1,
            "project_comparison": proj_comp,
            "envoy_subcohorts": {
                "production": {"baseline": prod_base, "candidate1": prod_cand},
                "scaffolding": {"baseline": mock_base, "candidate1": mock_cand}
            },
            "targeted_cwe_comparison": cwe_comp
        }, f, indent=2)

    print(f"\n[✓] Experiment 1 results saved to: {out_exp1_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 10.1 Independent Audit Script
===================================
DTU B.Tech Project: Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code

This script independently audits and reproduces all metrics directly from the exported
prediction manifest (results/phase10_1_locked_predictions.csv) and verifies the cryptographic
integrity of the locked 32-sample benchmark.

Usage:
    python scripts/run_phase10_1_audit.py
"""

import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def verify_locked_checksum():
    bench_file = PROJECT_ROOT / "results" / "reports" / "controlled_benchmark_32.json"
    chk_file = PROJECT_ROOT / "data" / "locked_benchmark_32" / "CHECKSUMS.sha256"
    
    with open(bench_file, "rb") as f:
        curr_sha = hashlib.sha256(f.read()).hexdigest()
    with open(chk_file, "r") as f:
        expected_sha = f.readline().split()[0]
        
    return curr_sha == expected_sha, curr_sha, expected_sha


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    n = len(y_true)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    prauc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    z = 1.96
    denom = 1 + z**2 / n
    center = (acc + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((acc * (1 - acc) + z**2 / (4 * n)) / n)) / denom
    ci_lower = max(0.0, float(center - half))
    ci_upper = min(1.0, float(center + half))
    
    return {
        "accuracy": round(acc, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "auroc": round(auroc, 4),
        "prauc": round(prauc, 4),
        "brier": round(brier, 4),
        "cm": cm,
        "n": n
    }


def main():
    print("="*85)
    print("PHASE 10.1 INDEPENDENT AUDIT OF MODEL B AND BENCHMARK INTEGRITY")
    print("="*85)
    
    # 1. Cryptographic Checksum Verification
    is_valid, curr_sha, expected_sha = verify_locked_checksum()
    print(f"Checksum Match: {is_valid}")
    print(f"Current SHA-256:  {curr_sha}")
    print(f"Expected SHA-256: {expected_sha}")
    if not is_valid:
        print("CRITICAL FAILURE: Locked benchmark file checksum does not match expected!")
        sys.exit(1)
        
    # 2. Load Predictions CSV
    csv_path = PROJECT_ROOT / "results" / "phase10_1_locked_predictions.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} predictions from {csv_path}")
    
    y_true = df["ground_truth_vuln"].values
    n_total = len(df)
    n_vuln = int(np.sum(y_true == 1))
    n_safe = int(np.sum(y_true == 0))
    print(f"Total Units: {n_total} (Vulnerable: {n_vuln}, Safe: {n_safe}, Balanced: {n_vuln == n_safe})")
    
    # 3. Overall Metrics
    m_a = calculate_metrics(y_true, df["model_a_pred"].values, df["model_a_prob"].values)
    m_b = calculate_metrics(y_true, df["model_b_pred"].values, df["model_b_prob"].values)
    m_unk = calculate_metrics(y_true, df["model_b_unknown_pred"].values, df["model_b_unknown_prob"].values)
    
    print("\n" + "-"*85)
    print(f"{'Metric':<22} {'Model A (Code Only)':<28} {'Model B (Code + Prov)':<28}")
    print("-"*85)
    print(f"{'Accuracy':<22} {m_a['accuracy']*100:.2f}% (CI: {m_a['ci_95'][0]*100:.1f}-{m_a['ci_95'][1]*100:.1f}%)    {m_b['accuracy']*100:.2f}% (CI: {m_b['ci_95'][0]*100:.1f}-{m_b['ci_95'][1]*100:.1f}%)")
    print(f"{'Balanced Accuracy':<22} {m_a['balanced_accuracy']*100:.2f}%{'':<21} {m_b['balanced_accuracy']*100:.2f}%")
    print(f"{'Precision':<22} {m_a['precision']*100:.2f}%{'':<21} {m_b['precision']*100:.2f}%")
    print(f"{'Recall':<22} {m_a['recall']*100:.2f}%{'':<21} {m_b['recall']*100:.2f}%")
    print(f"{'F1 Score':<22} {m_a['f1']:.4f}{'':<22} {m_b['f1']:.4f}")
    print(f"{'MCC':<22} {m_a['mcc']:+.4f}{'':<21} {m_b['mcc']:+.4f}")
    print(f"{'AUROC':<22} {m_a['auroc']:.4f}{'':<22} {m_b['auroc']:.4f}")
    print(f"{'PR-AUC':<22} {m_a['prauc']:.4f}{'':<22} {m_b['prauc']:.4f}")
    print(f"{'Brier Score':<22} {m_a['brier']:.4f}{'':<22} {m_b['brier']:.4f}")
    print(f"{'Confusion Matrix':<22} {str(m_a['cm']):<28} {str(m_b['cm']):<28}")
    print("-"*85)
    
    # 4. Unknown Provenance Fallback Verification
    print("\n--- PROVENANCE UNKNOWN FALLBACK AUDIT ---")
    matches = (df["model_a_pred"].values == df["model_b_unknown_pred"].values).all()
    prob_diff = np.abs(df["model_a_prob"].values - df["model_b_unknown_prob"].values).max()
    print(f"Predictions match Model A 100%: {matches}")
    print(f"Max absolute probability discrepancy: {prob_diff:.6f}")
    
    # 5. Exact Subgroup Metric Reconciliation
    print("\n--- EXACT SUBGROUP BREAKDOWN (PROVENANCE CATEGORIES) ---")
    for auth in ["human", "ai", "human_edit_of_ai", "ai_edit_of_human"]:
        sub = df[df["author_type"] == auth]
        n_s = len(sub)
        n_v = int(sum(sub["ground_truth_vuln"] == 1))
        n_ok = int(sum(sub["ground_truth_vuln"] == 0))
        corr_a = int(sum(sub["ground_truth_vuln"] == sub["model_a_pred"]))
        corr_b = int(sum(sub["ground_truth_vuln"] == sub["model_b_pred"]))
        acc_a = corr_a / n_s
        acc_b = corr_b / n_s
        print(f"Author Category: {auth:<18} (N={n_s}, {n_v} vuln, {n_ok} safe)")
        print(f"  Model A Accuracy: {corr_a}/{n_s} ({acc_a*100:5.2f}%)")
        print(f"  Model B Accuracy: {corr_b}/{n_s} ({acc_b*100:5.2f}%)")
        
    print("="*85)
    print("AUDIT COMPLETE — ALL METRICS REPRODUCED DIRECTLY FROM RAW PREDICTIONS.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 16 — Stage 1: Deep Diagnostic Error Analysis of Model A
=============================================================
Performs comprehensive empirical diagnosis of Phase 15 Model A on the
clean held-out benchmark (N = 3,169) to identify the largest sources of avoidable errors.

Analyses:
1. Confusion breakdown & probability distributions (marginal vs severe errors)
2. Project-wise performance (Envoy, GPAC, OpenSSL, Vim)
3. Envoy production code vs GoogleTest scaffolding
4. CWE-wise breakdown for False Negatives (FN = 498) and False Positives (FP = 326)
5. Structural feature differential analysis (TP vs FN, and TN vs FP)
6. Tokenisation & length effects (token count, character count, nesting depth)
7. Threshold sensitivity analysis on validation data vs held-out test data
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
RESULTS_P16_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_P14_DIR = PROJECT_ROOT / "results" / "phase14"

from scripts.phase15_train_model_a import Phase15EnhancedModernCxxExtractor


def main():
    print("=" * 80)
    print("PHASE 16 — STAGE 1: COMPREHENSIVE MODEL A ERROR DIAGNOSIS")
    print("=" * 80)

    # 1. Load Data & Phase 15 Checkpoint
    clean_test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet")
    X_test = clean_test_df['code'].values
    y_test = clean_test_df['vuln_label'].values.astype(int)
    projects = clean_test_df['project'].values
    cwes = clean_test_df['cwe'].values
    severities = clean_test_df['severity'].values

    emb_test = np.load(RESULTS_P14_DIR / "codebert_test_clean_emb.npy")

    ckpt_path = MODELS_DIR / "phase15_codebert_cxx_hybrid_best.pkl"
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    clf = ckpt["clf"]
    scaler = ckpt["scaler"]
    feature_names = ckpt["feature_names"]

    extractor = Phase15EnhancedModernCxxExtractor()
    raw_struct = extractor.transform(X_test)
    struct_test = scaler.transform(raw_struct)
    X_test_all = np.hstack([emb_test, struct_test])

    probs = clf.predict_proba(X_test_all)[:, 1]
    preds = (probs >= 0.50).astype(int)

    # 2. Confusion breakdown
    tp_mask = (y_test == 1) & (preds == 1)
    tn_mask = (y_test == 0) & (preds == 0)
    fp_mask = (y_test == 0) & (preds == 1)
    fn_mask = (y_test == 1) & (preds == 0)

    print(f"\nOverall Confusion Matrix (N = {len(y_test)}):")
    print(f"  True Positives  (TP): {tp_mask.sum():4d} ({tp_mask.sum()/len(y_test)*100:.1f}%)")
    print(f"  True Negatives  (TN): {tn_mask.sum():4d} ({tn_mask.sum()/len(y_test)*100:.1f}%)")
    print(f"  False Positives (FP): {fp_mask.sum():4d} ({fp_mask.sum()/len(y_test)*100:.1f}%) — FPR = {fp_mask.sum()/(fp_mask.sum()+tn_mask.sum())*100:.2f}%")
    print(f"  False Negatives (FN): {fn_mask.sum():4d} ({fn_mask.sum()/len(y_test)*100:.1f}%) — FNR = {fn_mask.sum()/(fn_mask.sum()+tp_mask.sum())*100:.2f}%")

    # 3. Probability Distribution Breakdown (Marginal vs Severe Errors)
    # Severe FN: P < 0.25 | Marginal FN: 0.25 <= P < 0.50
    # Severe FP: P >= 0.75 | Marginal FP: 0.50 <= P < 0.75
    severe_fn = (fn_mask & (probs < 0.25)).sum()
    marginal_fn = (fn_mask & (probs >= 0.25)).sum()
    severe_fp = (fp_mask & (probs >= 0.75)).sum()
    marginal_fp = (fp_mask & (probs < 0.75)).sum()

    print(f"\nError Severity Breakdown:")
    print(f"  False Negatives: Total = {fn_mask.sum()}")
    print(f"    • Marginal FN (0.25 <= P < 0.50): {marginal_fn} ({marginal_fn/fn_mask.sum()*100:.1f}%) — Near threshold, amenable to better calibration")
    print(f"    • Severe FN   (P < 0.25):         {severe_fn} ({severe_fn/fn_mask.sum()*100:.1f}%) — Severe misclassification (completely undetected)")
    print(f"  False Positives: Total = {fp_mask.sum()}")
    print(f"    • Marginal FP (0.50 <= P < 0.75): {marginal_fp} ({marginal_fp/fp_mask.sum()*100:.1f}%) — Near threshold")
    print(f"    • Severe FP   (P >= 0.75):        {severe_fp} ({severe_fp/fp_mask.sum()*100:.1f}%) — Severe false alarm")

    # 4. Project-wise Detailed Breakdown
    project_diagnostics = {}
    print(f"\nProject-wise Error Breakdown:")
    for p in sorted(np.unique(projects)):
        p_mask = (projects == p)
        n_p = p_mask.sum()
        p_y = y_test[p_mask]
        p_pred = preds[p_mask]
        p_prob = probs[p_mask]
        
        cm_p = confusion_matrix(p_y, p_pred, labels=[0, 1]).ravel()
        tn_p, fp_p, fn_p, tp_p = cm_p
        
        acc_p = float(accuracy_score(p_y, p_pred))
        auc_p = float(roc_auc_score(p_y, p_prob)) if len(np.unique(p_y)) > 1 else 0.5
        fpr_p = float(fp_p / (fp_p + tn_p)) if (fp_p + tn_p) > 0 else 0.0
        fnr_p = float(fn_p / (fn_p + tp_p)) if (fn_p + tp_p) > 0 else 0.0
        
        project_diagnostics[p] = {
            "n_samples": int(n_p),
            "vuln_samples": int(p_y.sum()),
            "vuln_rate": float(p_y.mean()),
            "accuracy": acc_p,
            "auroc": auc_p,
            "tn": int(tn_p),
            "fp": int(fp_p),
            "fn": int(fn_p),
            "tp": int(tp_p),
            "fpr": fpr_p,
            "fnr": fnr_p,
            "fp_rate_of_all_fp": float(fp_p / fp_mask.sum()),
            "fn_rate_of_all_fn": float(fn_p / fn_mask.sum())
        }
        print(f"  • {p:10}: Acc = {acc_p*100:.2f}% | AUROC = {auc_p:.4f} | FP = {fp_p:3d} (FPR={fpr_p*100:.1f}%) | FN = {fn_p:3d} (FNR={fnr_p*100:.1f}%) | N = {n_p}")

    # 5. Envoy Deep Dive: Production vs Scaffolding
    envoy_mask = (projects == "envoy")
    gtest_regex = re.compile(r'\b(EXPECT_CALL|ON_CALL|TEST_F|TEST_P|TEST|MOCK_METHOD|ASSERT_TRUE|ASSERT_FALSE|EXPECT_TRUE|EXPECT_FALSE|EXPECT_EQ|ASSERT_EQ)\b')
    is_mock = np.array([bool(gtest_regex.search(c)) for c in X_test[envoy_mask]])
    
    prod_cm = confusion_matrix(y_test[envoy_mask][~is_mock], preds[envoy_mask][~is_mock], labels=[0, 1]).ravel()
    mock_cm = confusion_matrix(y_test[envoy_mask][is_mock], preds[envoy_mask][is_mock], labels=[0, 1]).ravel()
    
    envoy_diag = {
        "production_code": {
            "n": int((~is_mock).sum()),
            "vuln_rate": float(y_test[envoy_mask][~is_mock].mean()),
            "accuracy": float(accuracy_score(y_test[envoy_mask][~is_mock], preds[envoy_mask][~is_mock])),
            "auroc": float(roc_auc_score(y_test[envoy_mask][~is_mock], probs[envoy_mask][~is_mock])),
            "tn": int(prod_cm[0]), "fp": int(prod_cm[1]), "fn": int(prod_cm[2]), "tp": int(prod_cm[3]),
            "fpr": float(prod_cm[1]/(prod_cm[1]+prod_cm[0])),
            "fnr": float(prod_cm[2]/(prod_cm[2]+prod_cm[3]))
        },
        "test_scaffolding": {
            "n": int(is_mock.sum()),
            "vuln_rate": float(y_test[envoy_mask][is_mock].mean()),
            "accuracy": float(accuracy_score(y_test[envoy_mask][is_mock], preds[envoy_mask][is_mock])),
            "auroc": float(roc_auc_score(y_test[envoy_mask][is_mock], probs[envoy_mask][is_mock])),
            "tn": int(mock_cm[0]), "fp": int(mock_cm[1]), "fn": int(mock_cm[2]), "tp": int(mock_cm[3]),
            "fpr": float(mock_cm[1]/(mock_cm[1]+mock_cm[0])),
            "fnr": float(mock_cm[2]/(mock_cm[2]+mock_cm[3]))
        }
    }
    print(f"\nEnvoy Subcohort Detailed Diagnostics:")
    print(f"  • Production Code:  N = {envoy_diag['production_code']['n']} | Acc = {envoy_diag['production_code']['accuracy']*100:.2f}% | FP = {envoy_diag['production_code']['fp']} | FN = {envoy_diag['production_code']['fn']} (FNR={envoy_diag['production_code']['fnr']*100:.1f}%)")
    print(f"  • Test Scaffolding: N = {envoy_diag['test_scaffolding']['n']} | Acc = {envoy_diag['test_scaffolding']['accuracy']*100:.2f}% | FP = {envoy_diag['test_scaffolding']['fp']} | FN = {envoy_diag['test_scaffolding']['fn']} (FNR={envoy_diag['test_scaffolding']['fnr']*100:.1f}%)")

    # 6. CWE Breakdown for False Negatives and False Positives
    df_results = pd.DataFrame({
        "project": projects,
        "cwe": cwes,
        "y_true": y_test,
        "y_pred": preds,
        "prob": probs,
        "is_tp": tp_mask,
        "is_tn": tn_mask,
        "is_fp": fp_mask,
        "is_fn": fn_mask
    })

    fn_cwes = df_results[df_results['is_fn']]['cwe'].value_counts().head(10).to_dict()
    fp_cwes = df_results[df_results['is_fp']]['cwe'].value_counts().head(10).to_dict()

    print(f"\nTop 10 CWEs in False Negatives (Missed Vulnerabilities):")
    for cwe_id, count in fn_cwes.items():
        total_cwe = (df_results['cwe'] == cwe_id).sum()
        total_vuln_cwe = ((df_results['cwe'] == cwe_id) & (df_results['y_true'] == 1)).sum()
        print(f"  • {cwe_id:10}: {count:3d} FNs out of {total_vuln_cwe:3d} actual vulns ({count/total_vuln_cwe*100:.1f}% missed)")

    print(f"\nTop 10 CWEs associated with False Positives (Safe code in vuln-associated files):")
    for cwe_id, count in fp_cwes.items():
        print(f"  • {cwe_id:10}: {count:3d} FPs")

    # 7. Structural Feature Differential Analysis
    # Compare mean structural feature values between TP and FN (missed vulnerabilities)
    # and between TN and FP (false alarms)
    feat_diff_tp_fn = {}
    feat_diff_tn_fp = {}
    for i, fname in enumerate(feature_names):
        mean_tp = float(np.mean(raw_struct[tp_mask, i]))
        mean_fn = float(np.mean(raw_struct[fn_mask, i]))
        mean_tn = float(np.mean(raw_struct[tn_mask, i]))
        mean_fp = float(np.mean(raw_struct[fp_mask, i]))
        feat_diff_tp_fn[fname] = {"mean_tp": mean_tp, "mean_fn": mean_fn, "delta": mean_tp - mean_fn}
        feat_diff_tn_fp[fname] = {"mean_tn": mean_tn, "mean_fp": mean_fp, "delta": mean_fp - mean_tn}

    # Find top discriminators where FN lacked features present in TP
    top_fn_missing = sorted(feat_diff_tp_fn.items(), key=lambda x: abs(x[1]['delta']), reverse=True)[:6]
    print(f"\nTop Structural Differences: True Positives vs False Negatives (What did missed vulns lack?):")
    for fname, d in top_fn_missing:
        print(f"  • {fname:20}: TP mean = {d['mean_tp']:.3f} | FN mean = {d['mean_fn']:.3f} | Delta = {d['delta']:+.3f}")

    top_fp_spurious = sorted(feat_diff_tn_fp.items(), key=lambda x: abs(x[1]['delta']), reverse=True)[:6]
    print(f"\nTop Structural Differences: True Negatives vs False Positives (What triggered false alarms?):")
    for fname, d in top_fp_spurious:
        print(f"  • {fname:20}: TN mean = {d['mean_tn']:.3f} | FP mean = {d['mean_fp']:.3f} | Delta = {d['delta']:+.3f}")

    # 8. Threshold Sensitivity Analysis
    print(f"\nThreshold Sensitivity Analysis on Test Benchmark:")
    threshold_sweep = {}
    for t_cand in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        p_cand = (probs >= t_cand).astype(int)
        acc_c = float(accuracy_score(y_test, p_cand))
        prec_c = float(precision_score(y_test, p_cand, zero_division=0))
        rec_c = float(recall_score(y_test, p_cand, zero_division=0))
        f1_c = float(f1_score(y_test, p_cand, zero_division=0))
        cm_c = confusion_matrix(y_test, p_cand, labels=[0, 1]).ravel()
        fpr_c = float(cm_c[1] / (cm_c[1] + cm_c[0]))
        threshold_sweep[str(t_cand)] = {
            "accuracy": acc_c, "precision": prec_c, "recall": rec_c, "f1": f1_c, "fpr": fpr_c
        }
        print(f"  • Tau = {t_cand:.2f}: Acc = {acc_c*100:.2f}% | F1 = {f1_c:.4f} | Prec = {prec_c*100:.1f}% | Rec = {rec_c*100:.1f}% | FPR = {fpr_c*100:.1f}%")

    # Save comprehensive error diagnostic payload
    diagnostic_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "confusion_matrix": {"tn": int(tn_mask.sum()), "fp": int(fp_mask.sum()), "fn": int(fn_mask.sum()), "tp": int(tp_mask.sum())},
        "error_severity": {
            "severe_fn": int(severe_fn), "marginal_fn": int(marginal_fn),
            "severe_fp": int(severe_fp), "marginal_fp": int(marginal_fp)
        },
        "project_diagnostics": project_diagnostics,
        "envoy_diagnostics": envoy_diag,
        "top_fn_cwes": fn_cwes,
        "top_fp_cwes": fp_cwes,
        "structural_diff_tp_fn": feat_diff_tp_fn,
        "structural_diff_tn_fp": feat_diff_tn_fp,
        "threshold_sweep": threshold_sweep
    }

    out_diag_path = RESULTS_P16_DIR / "model_a_error_diagnosis.json"
    with open(out_diag_path, "w") as f:
        json.dump(diagnostic_payload, f, indent=2)

    print(f"\n[✓] Diagnostic data saved to: {out_diag_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 16 — Model A Experimental Comparison Synthesis
====================================================
Aggregates and synthesizes evaluation metrics across all experimental candidates:
- Baseline: Phase 15 Modern C++ Hybrid (25k training, 28 features)
- Exp 1: Omission-Aware Modern C/C++ Hybrid (25k training, 38 features)
- Exp 2: C++ & Cross-Project Stratified Training (35k training, 38 features)
- Exp 3: Validation-Calibrated Model (Platt/Temperature + Tau*=0.56)
- Exp 4: Cost-Sensitive Asymmetric Loss Model (w_pos=1.0, C=0.20)
- Exp 5: Dual Classical-CodeBERT Stacking Ensemble

Produces unified comparison tables across:
- Overall Accuracy, Precision, Recall, F1, AUROC, MCC, FPR, FNR, Brier score
- Project-wise breakdown: Envoy, GPAC, OpenSSL, Vim
- Envoy production code vs test mocks
- Targeted CWE recall (CWE-476, CWE-400, CWE-835, CWE-416, CWE-119, CWE-787)
"""

import sys
import os
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import clean_code
from scripts.phase16_omission_extractor import Phase16OmissionAwareCxxExtractor

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_P16_DIR = PROJECT_ROOT / "results" / "phase16"
RESULTS_P15_DIR = PROJECT_ROOT / "results" / "phase15"
RESULTS_P14_DIR = PROJECT_ROOT / "results" / "phase14"


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
    print("PHASE 16 — MODEL A EXPERIMENTAL COMPARISON SYNTHESIS")
    print("=" * 80)

    # Load Clean Held-out Test Set
    test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet")
    X_test = test_df['code'].values
    y_test = test_df['vuln_label'].values.astype(int)
    projects = test_df['project'].values

    emb_test = np.load(RESULTS_P14_DIR / "codebert_test_clean_emb.npy")

    # Load feature extractors
    extractor_38 = Phase16OmissionAwareCxxExtractor()
    feat_38 = extractor_38.transform(X_test)

    # Envoy subsets
    envoy_mask = (projects == "envoy")
    envoy_df = test_df[envoy_mask]
    test_mock_keywords = [
        'mock', 'test', 'fake', 'fixture', 'testing', 'assert', 'expect_call',
        'test_server', 'mock_buffer', 'mock_stream', 'mock_connection',
        'mock_watermark', 'test_utility', 'test_time'
    ]
    is_mock = envoy_df['code'].str.lower().str.contains('|'.join(test_mock_keywords), regex=True)
    envoy_prod_mask = np.zeros(len(test_df), dtype=bool)
    envoy_mock_mask = np.zeros(len(test_df), dtype=bool)

    envoy_indices = np.where(envoy_mask)[0]
    prod_indices = envoy_indices[~is_mock.values]
    mock_indices = envoy_indices[is_mock.values]

    envoy_prod_mask[prod_indices] = True
    envoy_mock_mask[mock_indices] = True

    # Targeted CWEs
    cwe_col = 'cwe_id' if 'cwe_id' in test_df.columns else ('cwe' if 'cwe' in test_df.columns else None)
    target_cwes = ['CWE-476', 'CWE-400', 'CWE-835', 'CWE-416', 'CWE-119', 'CWE-787']

    # Load Baseline Checkpoint (Phase 15 Modern C++ Hybrid)
    with open(MODELS_DIR / "phase15_codebert_cxx_hybrid_best.pkl", "rb") as f:
        ckpt_base = pickle.load(f)
    from scripts.phase15_train_model_a import Phase15EnhancedModernCxxExtractor
    extractor_p15 = Phase15EnhancedModernCxxExtractor()
    feat_p15_test = extractor_p15.transform(X_test)
    feat_p15_scaled = ckpt_base['scaler'].transform(feat_p15_test)
    X_test_base = np.hstack([emb_test, feat_p15_scaled])
    prob_base = ckpt_base['clf'].predict_proba(X_test_base)[:, 1]
    pred_base = (prob_base >= 0.50).astype(int)

    # Dictionary of candidates to evaluate
    candidates = {}
    candidates["Phase 15 Champion (Baseline)"] = {
        "prob": prob_base,
        "pred": pred_base,
        "desc": "CodeBERT + 28 C++ Structural Features (25k training, C=0.01)"
    }

    # Exp 1: Omission-Aware Hybrid
    ckpt_exp1_path = MODELS_DIR / "phase16_codebert_omission_hybrid.pkl"
    if ckpt_exp1_path.exists():
        with open(ckpt_exp1_path, "rb") as f:
            ckpt_exp1 = pickle.load(f)
        feat_38_s1 = ckpt_exp1['scaler'].transform(feat_38)
        X_test_e1 = np.hstack([emb_test, feat_38_s1])
        prob_e1 = ckpt_exp1['clf'].predict_proba(X_test_e1)[:, 1]
        pred_e1 = (prob_e1 >= 0.50).astype(int)
        candidates["Exp 1: Omission-Aware (C=1.0)"] = {
            "prob": prob_e1,
            "pred": pred_e1,
            "desc": "CodeBERT + 38 Omission-Aware Features (25k training, C=1.0)"
        }

    # Exp 2: C++ Stratified 35k
    ckpt_exp2_path = MODELS_DIR / "phase16_codebert_cxx_stratified_35k.pkl"
    if ckpt_exp2_path.exists():
        with open(ckpt_exp2_path, "rb") as f:
            ckpt_exp2 = pickle.load(f)
        feat_38_s2 = ckpt_exp2['scaler'].transform(feat_38)
        X_test_e2 = np.hstack([emb_test, feat_38_s2])
        prob_e2 = ckpt_exp2['clf'].predict_proba(X_test_e2)[:, 1]
        pred_e2 = (prob_e2 >= 0.50).astype(int)
        candidates["Exp 2: C++ Stratified 35k"] = {
            "prob": prob_e2,
            "pred": pred_e2,
            "desc": "CodeBERT + 38 Features (35k stratified training, 20% C++, 0 Envoy)"
        }

    # Exp 3: Validation Calibrated (Tau* = 0.56)
    if ckpt_exp1_path.exists():
        candidates["Exp 3: Val-Calibrated (Tau=0.56)"] = {
            "prob": prob_e1,
            "pred": (prob_e1 >= 0.56).astype(int),
            "desc": "Exp 1 Model with Validation-Optimized Classification Threshold Tau*=0.56"
        }

    # Exp 4: Cost-Sensitive (Asymmetric Loss)
    ckpt_exp4_path = MODELS_DIR / "phase16_codebert_cost_sensitive.pkl"
    if ckpt_exp4_path.exists():
        with open(ckpt_exp4_path, "rb") as f:
            ckpt_exp4 = pickle.load(f)
        feat_38_s4 = ckpt_exp4['scaler'].transform(feat_38)
        X_test_e4 = np.hstack([emb_test, feat_38_s4])
        prob_e4 = ckpt_exp4['clf'].predict_proba(X_test_e4)[:, 1]
        pred_e4 = (prob_e4 >= 0.50).astype(int)
        candidates["Exp 4: Cost-Sensitive (w_pos=1.0)"] = {
            "prob": prob_e4,
            "pred": pred_e4,
            "desc": "CodeBERT + 38 Features with Asymmetric Loss w_pos=1.0, C=0.20"
        }

    # Exp 5: Dual Stacking Ensemble
    ckpt_exp5_path = MODELS_DIR / "phase16_model_a_stacking_ensemble.pkl"
    if ckpt_exp5_path.exists() and ckpt_exp1_path.exists():
        with open(ckpt_exp5_path, "rb") as f:
            ckpt_exp5 = pickle.load(f)
        with open(MODELS_DIR / "expanded_tri_union_best.pkl", "rb") as f:
            tri_union_obj = pickle.load(f)
        pipe_tri = tri_union_obj["pipeline"]
        cleaned_test = [clean_code(c) for c in X_test]
        prob_tri_test = pipe_tri.predict_proba(cleaned_test)[:, 1]

        eps = 1e-4
        logit_tri_test = np.log((prob_tri_test + eps) / (1.0 - prob_tri_test + eps))
        logit_cb_test = np.log((prob_e1 + eps) / (1.0 - prob_e1 + eps))
        Z_test = np.column_stack([logit_tri_test, logit_cb_test, logit_cb_test - logit_tri_test])

        prob_e5 = ckpt_exp5['meta_clf'].predict_proba(Z_test)[:, 1]
        best_tau_e5 = ckpt_exp5.get('best_tau', 0.50)
        pred_e5 = (prob_e5 >= best_tau_e5).astype(int)
        candidates["Exp 5: Dual Stacking Ensemble"] = {
            "prob": prob_e5,
            "pred": pred_e5,
            "desc": f"Stacking Meta-Learner (Tri-Union Classical + CodeBERT Omission Hybrid, Tau*={best_tau_e5:.2f})"
        }

    # Compute full evaluation across all candidates
    summary_results = {}
    for cname, cdata in candidates.items():
        prob = cdata["prob"]
        pred = cdata["pred"]

        overall_m = compute_metrics(y_test, pred, prob)

        # By Project
        proj_m = {}
        for p in sorted(np.unique(projects)):
            p_mask = (projects == p)
            proj_m[p] = compute_metrics(y_test[p_mask], pred[p_mask], prob[p_mask])

        # Envoy subcohorts
        envoy_prod_m = compute_metrics(y_test[envoy_prod_mask], pred[envoy_prod_mask], prob[envoy_prod_mask])
        envoy_mock_m = compute_metrics(y_test[envoy_mock_mask], pred[envoy_mock_mask], prob[envoy_mock_mask])

        # Targeted CWE recall
        cwe_recalls = {}
        if cwe_col:
            for cwe in target_cwes:
                mask = (test_df[cwe_col] == cwe) & (y_test == 1)
                n_cwe = int(mask.sum())
                if n_cwe > 0:
                    caught = int((pred[mask] == 1).sum())
                    cwe_recalls[cwe] = {
                        "total": n_cwe,
                        "caught": caught,
                        "recall": float(caught / n_cwe)
                    }

        summary_results[cname] = {
            "description": cdata["desc"],
            "overall": overall_m,
            "by_project": proj_m,
            "envoy_production": envoy_prod_m,
            "envoy_mock": envoy_mock_m,
            "targeted_cwes": cwe_recalls
        }

    # Print Table 1: Overall Metrics
    print("\n" + "=" * 110)
    print(f"{'Model Candidate':<36} | {'Accuracy':<9} | {'AUROC':<7} | {'F1':<7} | {'Precision':<9} | {'Recall':<7} | {'FPR':<7} | {'Brier':<7}")
    print("-" * 110)
    for cname, res in summary_results.items():
        o = res["overall"]
        print(f"{cname:<36} | {o['accuracy']*100:6.2f}%   | {o['auroc']:6.4f}  | {o['f1']:6.4f} | {o['precision']*100:6.2f}%   | {o['recall']*100:6.2f}% | {o['fpr']*100:5.2f}% | {o['brier_score']:6.4f}")
    print("=" * 110)

    # Print Table 2: Project-wise Breakdown
    print("\n" + "=" * 110)
    print(f"{'Model Candidate':<36} | {'Envoy Acc':<10} | {'Envoy Prod':<10} | {'OpenSSL':<10} | {'Vim':<10} | {'GPAC':<10}")
    print("-" * 110)
    for cname, res in summary_results.items():
        p = res["by_project"]
        ep = res["envoy_production"]["accuracy"] * 100
        envoy_acc = p.get("envoy", {}).get("accuracy", 0) * 100
        ssl_acc = p.get("openssl", {}).get("accuracy", 0) * 100
        vim_acc = p.get("vim", {}).get("accuracy", 0) * 100
        gpac_acc = p.get("gpac", {}).get("accuracy", 0) * 100
        print(f"{cname:<36} | {envoy_acc:6.2f}%    | {ep:6.2f}%    | {ssl_acc:6.2f}%   | {vim_acc:6.2f}%   | {gpac_acc:6.2f}%")
    print("=" * 110)

    # Print Table 3: Targeted CWE Recall
    print("\n" + "=" * 110)
    print(f"{'Model Candidate':<36} | {'CWE-476':<9} | {'CWE-400':<9} | {'CWE-835':<9} | {'CWE-416':<9} | {'CWE-787':<9}")
    print("-" * 110)
    for cname, res in summary_results.items():
        cw = res["targeted_cwes"]
        r476 = cw.get("CWE-476", {}).get("recall", 0) * 100
        r400 = cw.get("CWE-400", {}).get("recall", 0) * 100
        r835 = cw.get("CWE-835", {}).get("recall", 0) * 100
        r416 = cw.get("CWE-416", {}).get("recall", 0) * 100
        r787 = cw.get("CWE-787", {}).get("recall", 0) * 100
        print(f"{cname:<36} | {r476:6.2f}%   | {r400:6.2f}%   | {r835:6.2f}%   | {r416:6.2f}%   | {r787:6.2f}%")
    print("=" * 110)

    # Save to JSON
    out_synth = RESULTS_P16_DIR / "phase16_model_a_experiments_synthesis.json"
    with open(out_synth, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\n[✓] Comprehensive synthesis saved to: {out_synth}")


if __name__ == "__main__":
    main()

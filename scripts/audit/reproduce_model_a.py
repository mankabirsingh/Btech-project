#!/usr/bin/env python3
"""
scripts/audit/reproduce_model_a.py
==================================
Independent reproducibility and validity audit of Model A.
Evaluates all Model A checkpoints on vulnerability_test_heldout_projects_clean.parquet.
Computes:
- Full metrics (Acc, Majority Baseline, Bal Acc, MCC, AUROC, AUPRC, F1, P, R, FPR, Brier)
  with 95% stratified bootstrap CIs (1000 resamples, seed 42) overall and per project.
- Per-CWE recall for CWE-476, 400, 835, 416, 787, 119 with Wilson 95% CIs.
- Paired McNemar tests (overall and Envoy) and DeLong AUROC tests against Phase 15 Champion.
"""

import sys
import os
import re
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_P14_DIR = PROJECT_ROOT / "results" / "phase14"
RESULTS_AUDIT_DIR = PROJECT_ROOT / "results" / "audit"
RESULTS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

from scripts.phase15_train_model_a import Phase15EnhancedModernCxxExtractor
from scripts.phase16_omission_extractor import Phase16OmissionAwareCxxExtractor
from src.data.preprocessing import clean_code


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = k / n
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denom
    half_width = (z * np.sqrt((p_hat * (1.0 - p_hat) / n) + ((z**2) / (4.0 * (n**2))))) / denom
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42
) -> Dict[str, List[float]]:
    """Computes 95% stratified bootstrap CIs for Accuracy, MCC, AUROC, AUPRC."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    rng = np.random.RandomState(seed)
    accs, mccs, aurocs, auprcs = [], [], [], []

    for _ in range(n_resamples):
        b_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        b_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        y_b = y_true[b_idx]
        pred_b = pred[b_idx]
        prob_b = prob[b_idx]

        accs.append(accuracy_score(y_b, pred_b))
        mccs.append(matthews_corrcoef(y_b, pred_b) if len(np.unique(pred_b)) > 1 else 0.0)
        aurocs.append(roc_auc_score(y_b, prob_b) if len(np.unique(y_b)) > 1 else 0.5)
        auprcs.append(average_precision_score(y_b, prob_b) if len(np.unique(y_b)) > 1 else np.mean(y_b))

    return {
        "accuracy_ci_95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "mcc_ci_95": [float(np.percentile(mccs, 2.5)), float(np.percentile(mccs, 97.5))],
        "auroc_ci_95": [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))],
        "auprc_ci_95": [float(np.percentile(auprcs, 2.5)), float(np.percentile(auprcs, 97.5))]
    }


def mcnemar_test(y_true: np.ndarray, pred_champ: np.ndarray, pred_cand: np.ndarray) -> Dict[str, Any]:
    """Computes paired McNemar test comparing Champion vs Candidate predictions."""
    corr_champ = (pred_champ == y_true)
    corr_cand = (pred_cand == y_true)

    a = int(np.sum(corr_champ & corr_cand))
    b = int(np.sum(corr_champ & (~corr_cand)))  # Champion correct, Candidate wrong
    c = int(np.sum((~corr_champ) & corr_cand))  # Candidate correct, Champion wrong
    d = int(np.sum((~corr_champ) & (~corr_cand)))

    if (b + c) > 0:
        # Edwards continuity correction
        chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_val = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = 0.0
        p_val = 1.0

    return {
        "contingency_table": {"a_both_correct": a, "b_champ_only": b, "c_cand_only": c, "d_both_wrong": d},
        "chi2": float(chi2),
        "p_value": float(p_val),
        "significant_p05": bool(p_val < 0.05)
    }


def delong_roc_test(y_true: np.ndarray, prob_champ: np.ndarray, prob_cand: np.ndarray) -> Dict[str, Any]:
    """Computes DeLong test for paired AUROCs."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    d1 = prob_champ[pos_idx][:, None] - prob_champ[neg_idx][None, :]
    psi1 = (d1 > 0).astype(float) + 0.5 * (d1 == 0).astype(float)
    v10 = np.mean(psi1, axis=1)
    v01 = np.mean(psi1, axis=0)
    auc1 = float(np.mean(v10))

    d2 = prob_cand[pos_idx][:, None] - prob_cand[neg_idx][None, :]
    psi2 = (d2 > 0).astype(float) + 0.5 * (d2 == 0).astype(float)
    v20 = np.mean(psi2, axis=1)
    v02 = np.mean(psi2, axis=0)
    auc2 = float(np.mean(v20))

    s10 = np.cov(np.vstack([v10, v20]))
    s01 = np.cov(np.vstack([v01, v02]))
    s = s10 / m + s01 / n

    var_diff = s[0, 0] + s[1, 1] - 2.0 * s[0, 1]
    diff = auc1 - auc2
    if var_diff <= 0:
        z = 0.0
        p_val = 1.0
    else:
        z = diff / np.sqrt(var_diff)
        p_val = float(2.0 * stats.norm.sf(abs(z)))

    return {
        "auc_champ": auc1,
        "auc_cand": auc2,
        "delta": float(auc2 - auc1),
        "z_score": float(z),
        "p_value": float(p_val),
        "significant_p05": bool(p_val < 0.05)
    }


def compute_cohort_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    compute_ci: bool = True
) -> Dict[str, Any]:
    """Computes full classification and calibration metrics for a cohort."""
    n = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    prevalence = float(n_pos / n) if n > 0 else 0.0
    majority_baseline = float(max(n_pos, n_neg) / n) if n > 0 else 0.0

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    acc = float(accuracy_score(y_true, pred))
    bal_acc = float(balanced_accuracy_score(y_true, pred))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0

    try:
        auroc = float(roc_auc_score(y_true, prob))
    except Exception:
        auroc = 0.50
    try:
        auprc = float(average_precision_score(y_true, prob))
    except Exception:
        auprc = prevalence

    brier = float(brier_score_loss(y_true, prob))

    res = {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": prevalence,
        "majority_class_baseline": majority_baseline,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "auroc": auroc,
        "auprc": auprc,
        "fpr": fpr,
        "fnr": fnr,
        "brier_score": brier
    }

    if compute_ci and n_pos > 0 and n_neg > 0:
        cis = stratified_bootstrap_ci(y_true, pred, prob, n_resamples=1000, seed=42)
        res.update(cis)
    else:
        res["accuracy_ci_95"] = [acc, acc]
        res["mcc_ci_95"] = [mcc, mcc]
        res["auroc_ci_95"] = [auroc, auroc]
        res["auprc_ci_95"] = [auprc, auprc]

    return res


def main():
    t_start = time.time()
    print("=" * 80)
    print("AUDIT: MODEL A INDEPENDENT REPRODUCIBILITY & VALIDITY EVALUATION")
    print("=" * 80)

    # 1. Load Clean Held-Out Test Set
    test_path = SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet"
    print(f"\n[1] Loading Clean Held-Out Test Set: {test_path}")
    test_df = pd.read_parquet(test_path)
    X_test = test_df['code'].values
    y_test = test_df['vuln_label'].values.astype(int)
    projects = test_df['project'].values

    # Checkpoint cached CodeBERT embeddings
    emb_path = RESULTS_P14_DIR / "codebert_test_clean_emb.npy"
    print(f"  Loading cached CodeBERT test embeddings: {emb_path.name}")
    emb_test = np.load(emb_path)

    # 2. Extract structural features
    print("\n[2] Extracting C/C++ Structural Features for Test Snippets...")
    t0 = time.time()
    ext_28 = Phase15EnhancedModernCxxExtractor()
    feat_28 = ext_28.transform(X_test)

    ext_38 = Phase16OmissionAwareCxxExtractor()
    feat_38 = ext_38.transform(X_test)
    print(f"  Extracted 28-dim and 38-dim features in {time.time()-t0:.2f}s")

    # 3. Cohort Masks
    # Non-mock definition for Envoy production code: gtest regex (as in phase15 & exp1)
    gtest_regex = re.compile(r'\b(EXPECT_CALL|ON_CALL|TEST_F|TEST_P|TEST|MOCK_METHOD|ASSERT_TRUE|ASSERT_FALSE|EXPECT_TRUE|EXPECT_FALSE|EXPECT_EQ|ASSERT_EQ)\b')
    is_mock = np.array([bool(gtest_regex.search(c)) for c in X_test])

    cohorts = {
        "Overall": np.ones(len(y_test), dtype=bool),
        "Envoy": (projects == "envoy"),
        "Envoy Production (non-mock)": (projects == "envoy") & (~is_mock),
        "OpenSSL": (projects == "openssl"),
        "Vim": (projects == "vim"),
        "GPAC": (projects == "gpac")
    }

    # 4. Reload and Predict Every Model A Checkpoint
    print("\n[3] Reloading Checkpoints and Computing Test Predictions...")

    # Checkpoint 1: Phase 15 Champion
    with open(MODELS_DIR / "phase15_codebert_cxx_hybrid_best.pkl", "rb") as f:
        ckpt_p15 = pickle.load(f)
    s_p15 = ckpt_p15['scaler'].transform(feat_28)
    X_p15 = np.hstack([emb_test, s_p15])
    prob_p15 = ckpt_p15['clf'].predict_proba(X_p15)[:, 1]
    pred_p15 = (prob_p15 >= 0.50).astype(int)

    # Checkpoint 2: Phase 16 Exp 1 (Omission-Aware, C=1.0)
    with open(MODELS_DIR / "phase16_codebert_omission_hybrid.pkl", "rb") as f:
        ckpt_e1 = pickle.load(f)
    s_e1 = ckpt_e1['scaler'].transform(feat_38)
    X_e1 = np.hstack([emb_test, s_e1])
    prob_e1 = ckpt_e1['clf'].predict_proba(X_e1)[:, 1]
    pred_e1 = (prob_e1 >= 0.50).astype(int)

    # Checkpoint 3: Phase 16 Exp 2 (35k C++ Stratified)
    with open(MODELS_DIR / "phase16_codebert_cxx_stratified_35k.pkl", "rb") as f:
        ckpt_e2 = pickle.load(f)
    s_e2 = ckpt_e2['scaler'].transform(feat_38)
    X_e2 = np.hstack([emb_test, s_e2])
    prob_e2 = ckpt_e2['clf'].predict_proba(X_e2)[:, 1]
    pred_e2 = (prob_e2 >= 0.50).astype(int)

    # Checkpoint 4: Phase 16 Exp 4 (Cost-Sensitive)
    with open(MODELS_DIR / "phase16_codebert_cost_sensitive.pkl", "rb") as f:
        ckpt_e4 = pickle.load(f)
    s_e4 = ckpt_e4['scaler'].transform(feat_38)
    X_e4 = np.hstack([emb_test, s_e4])
    prob_e4 = ckpt_e4['clf'].predict_proba(X_e4)[:, 1]
    pred_e4 = (prob_e4 >= 0.50).astype(int)

    # Checkpoint 5: Phase 16 Exp 5 (Stacking Ensemble)
    with open(MODELS_DIR / "phase16_model_a_stacking_ensemble.pkl", "rb") as f:
        ckpt_e5 = pickle.load(f)
    with open(MODELS_DIR / "expanded_tri_union_best.pkl", "rb") as f:
        tri_obj = pickle.load(f)
    cleaned_test = [clean_code(c) for c in X_test]
    prob_tri = tri_obj['pipeline'].predict_proba(cleaned_test)[:, 1]
    eps = 1e-4
    logit_tri = np.log((prob_tri + eps) / (1.0 - prob_tri + eps))
    logit_cb = np.log((prob_e1 + eps) / (1.0 - prob_e1 + eps))
    Z_test = np.column_stack([logit_tri, logit_cb, logit_cb - logit_tri])
    prob_e5 = ckpt_e5['meta_clf'].predict_proba(Z_test)[:, 1]
    tau_e5 = ckpt_e5.get('best_tau', 0.50)
    pred_e5 = (prob_e5 >= tau_e5).astype(int)

    # Setup 6: Exp 3 Threshold Setup (Applied to Exp 1 as verified in A4a)
    prob_e3 = prob_e1
    pred_e3 = (prob_e1 >= 0.560).astype(int)

    # Also include Phase 15 Champion with Tau=0.560 for transparent comparison
    prob_p15_tau56 = prob_p15
    pred_p15_tau56 = (prob_p15 >= 0.560).astype(int)

    models = {
        "Phase 15 Champion": {
            "checkpoint": "phase15_codebert_cxx_hybrid_best.pkl",
            "threshold": 0.50,
            "prob": prob_p15,
            "pred": pred_p15
        },
        "Exp 1: Omission-Aware (C=1.0)": {
            "checkpoint": "phase16_codebert_omission_hybrid.pkl",
            "threshold": 0.50,
            "prob": prob_e1,
            "pred": pred_e1
        },
        "Exp 2: C++ Stratified 35k": {
            "checkpoint": "phase16_codebert_cxx_stratified_35k.pkl",
            "threshold": 0.50,
            "prob": prob_e2,
            "pred": pred_e2
        },
        "Exp 3: Val-Calibrated (Exp 1 @ Tau=0.56)": {
            "checkpoint": "phase16_codebert_omission_hybrid.pkl",
            "threshold": 0.56,
            "prob": prob_e3,
            "pred": pred_e3
        },
        "Exp 4: Cost-Sensitive": {
            "checkpoint": "phase16_codebert_cost_sensitive.pkl",
            "threshold": 0.50,
            "prob": prob_e4,
            "pred": pred_e4
        },
        "Exp 5: Dual Stacking Ensemble": {
            "checkpoint": "phase16_model_a_stacking_ensemble.pkl",
            "threshold": tau_e5,
            "prob": prob_e5,
            "pred": pred_e5
        },
        "Phase 15 Champion @ Tau=0.56": {
            "checkpoint": "phase15_codebert_cxx_hybrid_best.pkl",
            "threshold": 0.56,
            "prob": prob_p15_tau56,
            "pred": pred_p15_tau56
        }
    }

    # 5. Full Metrics Computation with 95% Bootstrap CIs
    print("\n[4] Computing Full Metrics & Stratified Bootstrap CIs (1000 resamples)...")
    model_metrics = {}
    for mname, mdata in models.items():
        print(f"  Evaluating {mname}...")
        m_cohorts = {}
        for cname, cmask in cohorts.items():
            y_sub = y_test[cmask]
            pred_sub = mdata["pred"][cmask]
            prob_sub = mdata["prob"][cmask]
            m_cohorts[cname] = compute_cohort_metrics(y_sub, pred_sub, prob_sub, compute_ci=True)
        model_metrics[mname] = {
            "checkpoint": mdata["checkpoint"],
            "threshold": mdata["threshold"],
            "cohorts": m_cohorts
        }

    # 6. Per-CWE Recall Analysis (CWE-476, 400, 835, 416, 787, 119)
    print("\n[5] Computing Per-CWE Recalls with Wilson 95% CIs...")
    target_cwes = ["CWE-476", "CWE-400", "CWE-835", "CWE-416", "CWE-787", "CWE-119"]
    cwe_results = {}

    for cwe in target_cwes:
        cwe_mask = (test_df['cwe'] == cwe) & (y_test == 1)
        n_cwe = int(np.sum(cwe_mask))
        cwe_results[cwe] = {"n_total_vulnerable": n_cwe, "models": {}}

        for mname, mdata in models.items():
            k_caught = int(np.sum(mdata["pred"][cwe_mask] == 1)) if n_cwe > 0 else 0
            recall = float(k_caught / n_cwe) if n_cwe > 0 else 0.0
            ci_low, ci_high = wilson_ci(k_caught, n_cwe, confidence=0.95)
            cwe_results[cwe]["models"][mname] = {
                "k_caught": k_caught,
                "n_total": n_cwe,
                "recall": recall,
                "wilson_ci_95": [ci_low, ci_high]
            }

    # 7. Paired Statistical Tests against Phase 15 Champion
    print("\n[6] Computing Paired Statistical Tests against Phase 15 Champion...")
    paired_tests = {}
    envoy_mask = cohorts["Envoy"]

    for mname, mdata in models.items():
        if mname == "Phase 15 Champion":
            continue
        # McNemar Overall
        mcnemar_ov = mcnemar_test(y_test, pred_p15, mdata["pred"])
        # McNemar Envoy
        mcnemar_env = mcnemar_test(y_test[envoy_mask], pred_p15[envoy_mask], mdata["pred"][envoy_mask])
        # DeLong AUROC Overall
        delong_ov = delong_roc_test(y_test, prob_p15, mdata["prob"])

        paired_tests[mname] = {
            "mcnemar_overall": mcnemar_ov,
            "mcnemar_envoy": mcnemar_env,
            "delong_overall": delong_ov
        }

    # 8. Save Audit Results JSON
    total_runtime = time.time() - t_start
    output_payload = {
        "metadata": {
            "task": "Model A Reproducibility and Validity Audit",
            "clean_test_path": str(test_path),
            "n_samples": len(test_df),
            "runtime_seconds": round(total_runtime, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "models_evaluated": list(models.keys()),
        "full_metrics": model_metrics,
        "per_cwe_recall": cwe_results,
        "paired_tests": paired_tests
    }

    out_file = RESULTS_AUDIT_DIR / "model_a_reproduction.json"
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[✓] Audit successfully completed in {total_runtime:.2f}s!")
    print(f"[✓] Results written to: {out_file}")


if __name__ == "__main__":
    main()

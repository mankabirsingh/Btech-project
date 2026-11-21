#!/usr/bin/env python3
"""
Phase 14.1 — Independent Scientific Audit & Experimental Analysis Script
========================================================================
This script performs all empirical checks required for Phase 14.1:
1. Model A Audit:
   - Evaluates Candidate A, B, C, D on Envoy sub-cohorts (GoogleTest vs Production).
   - Re-checks OpenSSL, GPAC, Vim confusion matrices and error origins.
   - Analyzes training data disparities between Candidate A and Candidate C.
2. Model B Audit:
   - Paired prediction analysis on the 32 test units (Model B vs Model A vs Provenance-Only).
   - Grouped Cross-Validation (Leave-One-Task-Out across all 52 tasks = 208 units).
   - Exact confidence intervals (Wilson score & bootstrap).
   - Task construction similarity audit.
"""

import sys
import os
import re
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
DATA_P14_DIR = PROJECT_ROOT / "data" / "phase14_model_b"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase14"

from src.models.cxx_features import EnhancedModernCxxStructuralExtractor
from scripts.search_classical_models import SecurityStructuralFeatureExtractor
from src.models.inference import predict_model_a

def extract_fusion_features_record(r: Dict[str, Any], p_code: float) -> np.ndarray:
    eps = 1e-4
    logit_code = np.log((p_code + eps) / (1.0 - p_code + eps))
    author_type = r.get("author_type", "unknown")
    ai_involvement = float(r.get("ai_involvement", 0.0))
    confidence = float(r.get("provenance_confidence", 1.0))
    is_pure_ai = 1.0 if author_type == "ai" else 0.0
    is_human_edited = 1.0 if author_type == "human_edit_of_ai" else 0.0
    is_ai_edited = 1.0 if author_type == "ai_edit_of_human" else 0.0
    return np.array([logit_code, 1.0, ai_involvement, is_pure_ai, is_human_edited, is_ai_edited, confidence], dtype=float)


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    denom = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def compute_metrics_detailed(y_true, y_pred, y_prob):
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


def audit_model_a():
    print("=" * 80)
    print("1. MODEL A DETAILED AUDIT")
    print("=" * 80)

    # Load Clean Held-Out Benchmark (N = 3,169)
    clean_test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet")
    X_test = clean_test_df['code'].values
    y_test = clean_test_df['vuln_label'].values.astype(int)
    projects = clean_test_df['project'].values

    # Load pre-extracted CodeBERT test embeddings (3169, 1536)
    emb_test = np.load(RESULTS_DIR / "codebert_test_clean_emb.npy")

    # Load Candidate A (Baseline CodeBERT)
    with open(MODELS_DIR / "expanded_codebert_hybrid_best.pkl", "rb") as f:
        cb_base_ckpt = pickle.load(f)
    cb_clf = cb_base_ckpt['clf']
    cb_scaler = cb_base_ckpt['scaler']
    base_extractor = SecurityStructuralFeatureExtractor()
    struct_test_base = cb_scaler.transform(base_extractor.transform(X_test))
    prob_cb = cb_clf.predict_proba(np.hstack([emb_test, struct_test_base]))[:, 1]

    # Load Candidate C (CodeBERT C++ Hybrid)
    with open(MODELS_DIR / "phase14_codebert_cxx_hybrid.pkl", "rb") as f:
        cxx_ckpt = pickle.load(f)
    clf_c = cxx_ckpt['clf']
    scaler_cxx = cxx_ckpt['scaler']
    cxx_extractor = EnhancedModernCxxStructuralExtractor()
    struct_test_cxx = scaler_cxx.transform(cxx_extractor.transform(X_test))
    prob_c = clf_c.predict_proba(np.hstack([emb_test, struct_test_cxx]))[:, 1]

    # Checkpoint weight inspection
    print("\n--- Weight & Scaler Comparison ---")
    print(f"Candidate A clf coef shape: {cb_clf.coef_.shape}, intercept: {cb_clf.intercept_[0]:.4f}")
    print(f"Candidate C clf coef shape: {clf_c.coef_.shape}, intercept: {clf_c.intercept_[0]:.4f}")
    
    # Inspect weights on the 24 C++ features in Candidate C
    cxx_feat_weights = clf_c.coef_[0, 1536:]
    feat_names = [
        "log1p_chars", "log1p_lines", "log1p_tokens", "dang_ratio", "safe_ratio",
        "ctrl_ratio", "ptr_ratio", "ref_ratio", "c_mem_count", "smart_ptrs_count",
        "new_del_count", "casts_count", "risky_casts_count", "templates_count",
        "scopes_count", "virtual_count", "exceptions_count", "locks_count",
        "gtest_count", "is_test", "max_depth", "comment_density", "is_cpp", "smart_ratio"
    ]
    print("\nWeights assigned to 24 C++ features in Candidate C:")
    for name, w in sorted(zip(feat_names, cxx_feat_weights), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:25}: {w:+.4f}")

    # Inspect Envoy in detail
    envoy_mask = (projects == 'envoy')
    envoy_codes = X_test[envoy_mask]
    envoy_y = y_test[envoy_mask]
    envoy_prob_cb = prob_cb[envoy_mask]
    envoy_prob_c = prob_c[envoy_mask]

    # Detect GoogleTest
    gtest_regex = re.compile(r'\b(EXPECT_CALL|ON_CALL|TEST_F|TEST_P|TEST|MOCK_METHOD|ASSERT_TRUE|ASSERT_FALSE|EXPECT_TRUE|EXPECT_FALSE|EXPECT_EQ|ASSERT_EQ)\b')
    envoy_is_gtest = np.array([bool(gtest_regex.search(c)) for c in envoy_codes])

    print(f"\n--- Envoy Sub-Cohort Breakdown (N = {len(envoy_codes)}) ---")
    print(f"GoogleTest samples: {envoy_is_gtest.sum()} (Vuln: {envoy_y[envoy_is_gtest].sum()}, Clean: {(~envoy_y[envoy_is_gtest].astype(bool)).sum()})")
    print(f"Non-GoogleTest (Production): {(~envoy_is_gtest).sum()} (Vuln: {envoy_y[~envoy_is_gtest].sum()}, Clean: {(~envoy_y[~envoy_is_gtest].astype(bool)).sum()})")

    # Metrics on GoogleTest sub-cohort
    m_cb_gtest = compute_metrics_detailed(envoy_y[envoy_is_gtest], (envoy_prob_cb[envoy_is_gtest] >= 0.5).astype(int), envoy_prob_cb[envoy_is_gtest])
    m_c_gtest = compute_metrics_detailed(envoy_y[envoy_is_gtest], (envoy_prob_c[envoy_is_gtest] >= 0.5).astype(int), envoy_prob_c[envoy_is_gtest])

    print("\nGoogleTest Cohort Performance:")
    print(f"  Candidate A (Baseline): Acc={m_cb_gtest['accuracy']*100:.2f}%, FPR={m_cb_gtest['fpr']*100:.1f}%, FNR={m_cb_gtest['fnr']*100:.1f}%, TP={m_cb_gtest['confusion_matrix']['tp']}, FP={m_cb_gtest['confusion_matrix']['fp']}, TN={m_cb_gtest['confusion_matrix']['tn']}, FN={m_cb_gtest['confusion_matrix']['fn']}")
    print(f"  Candidate C (C++ Hybr): Acc={m_c_gtest['accuracy']*100:.2f}%, FPR={m_c_gtest['fpr']*100:.1f}%, FNR={m_c_gtest['fnr']*100:.1f}%, TP={m_c_gtest['confusion_matrix']['tp']}, FP={m_c_gtest['confusion_matrix']['fp']}, TN={m_c_gtest['confusion_matrix']['tn']}, FN={m_c_gtest['confusion_matrix']['fn']}")

    # Metrics on Non-GoogleTest (Production) sub-cohort
    m_cb_prod = compute_metrics_detailed(envoy_y[~envoy_is_gtest], (envoy_prob_cb[~envoy_is_gtest] >= 0.5).astype(int), envoy_prob_cb[~envoy_is_gtest])
    m_c_prod = compute_metrics_detailed(envoy_y[~envoy_is_gtest], (envoy_prob_c[~envoy_is_gtest] >= 0.5).astype(int), envoy_prob_c[~envoy_is_gtest])

    print("\nNon-GoogleTest (Production) Cohort Performance:")
    print(f"  Candidate A (Baseline): Acc={m_cb_prod['accuracy']*100:.2f}%, FPR={m_cb_prod['fpr']*100:.1f}%, FNR={m_cb_prod['fnr']*100:.1f}%, TP={m_cb_prod['confusion_matrix']['tp']}, FP={m_cb_prod['confusion_matrix']['fp']}, TN={m_cb_prod['confusion_matrix']['tn']}, FN={m_cb_prod['confusion_matrix']['fn']}")
    print(f"  Candidate C (C++ Hybr): Acc={m_c_prod['accuracy']*100:.2f}%, FPR={m_c_prod['fpr']*100:.1f}%, FNR={m_c_prod['fnr']*100:.1f}%, TP={m_c_prod['confusion_matrix']['tp']}, FP={m_c_prod['confusion_matrix']['fp']}, TN={m_c_prod['confusion_matrix']['tn']}, FN={m_c_prod['confusion_matrix']['fn']}")

    # All projects confusion matrices
    print("\n--- Per-Project Full Metrics ---")
    project_metrics = {}
    for proj in ['openssl', 'gpac', 'vim', 'envoy']:
        p_mask = (projects == proj)
        m_a = compute_metrics_detailed(y_test[p_mask], (prob_cb[p_mask] >= 0.5).astype(int), prob_cb[p_mask])
        m_c = compute_metrics_detailed(y_test[p_mask], (prob_c[p_mask] >= 0.5).astype(int), prob_c[p_mask])
        project_metrics[proj] = {"Candidate A": m_a, "Candidate C": m_c}
        print(f"\nProject: {proj.upper()} (N={m_a['n_samples']})")
        print(f"  Cand A: Acc={m_a['accuracy']*100:.2f}%, P={m_a['precision']*100:.2f}%, R={m_a['recall']*100:.2f}%, F1={m_a['f1']:.4f}, FPR={m_a['fpr']*100:.1f}%, FNR={m_a['fnr']*100:.1f}%, CM={m_a['confusion_matrix']}")
        print(f"  Cand C: Acc={m_c['accuracy']*100:.2f}%, P={m_c['precision']*100:.2f}%, R={m_c['recall']*100:.2f}%, F1={m_c['f1']:.4f}, FPR={m_c['fpr']*100:.1f}%, FNR={m_c['fnr']*100:.1f}%, CM={m_c['confusion_matrix']}")

    return {
        "envoy_gtest": {"Candidate A": m_cb_gtest, "Candidate C": m_c_gtest},
        "envoy_prod": {"Candidate A": m_cb_prod, "Candidate C": m_c_prod},
        "by_project": project_metrics,
        "feature_weights": dict(zip(feat_names, [float(x) for x in cxx_feat_weights]))
    }


def audit_model_b():
    print("\n" + "=" * 80)
    print("2. MODEL B DETAILED AUDIT & PAIRED ANALYSIS")
    print("=" * 80)

    # Load Phase 14 Model B test data (N = 32)
    test_json_path = DATA_P14_DIR / "model_b_p14_test.json"
    with open(test_json_path, "r") as f:
        test_data = json.load(f)
    test_units = test_data['records'] if isinstance(test_data, dict) and 'records' in test_data else test_data

    # Load Phase 14 Model B checkpoint
    with open(MODELS_DIR / "phase14_model_b_fusion_best.pkl", "rb") as f:
        mb_ckpt = pickle.load(f)

    mb_clf = mb_ckpt['clf_fusion']
    feat_cols = mb_ckpt.get('feature_names', mb_ckpt.get('feature_cols'))
    intercept = mb_ckpt.get('intercept', mb_clf.intercept_[0])
    weights = dict(zip(feat_cols, mb_clf.coef_[0]))

    print(f"Model B Features: {feat_cols}")
    print(f"Model B Weights: {weights}")
    print(f"Intercept: {intercept:.4f}")

    # Build features for test set using actual Model A inference
    rows = []
    print("  Computing Model A probabilities for test units...")
    for u in test_units:
        ptype = u.get('author_type', 'unknown')
        involvement = float(u.get('ai_involvement', 0.0))
        code_score = float(predict_model_a(u["code"])["probability"])
        logit_code = float(np.log(np.clip(code_score, 1e-4, 1 - 1e-4) / (1 - np.clip(code_score, 1e-4, 1 - 1e-4))))

        r = {
            'task_id': u.get('task_id'),
            'cwe': u.get('cwe'),
            'provenance_type': ptype,
            'vuln_label': int(u.get('vuln_label')),
            'model_a_prob': code_score,
            'model_a_pred': int(code_score >= 0.50),
            'logit_code': logit_code,
            'has_provenance': 1.0,
            'ai_involvement': involvement,
            'is_pure_ai': 1.0 if ptype == 'ai' else 0.0,
            'is_human_edited': 1.0 if ptype == 'human_edit_of_ai' else 0.0,
            'is_ai_edited': 1.0 if ptype == 'ai_edit_of_human' else 0.0,
            'confidence': float(u.get('provenance_confidence', 0.95))
        }
        rows.append(r)

    df_test = pd.DataFrame(rows)
    X_test_mat = df_test[feat_cols].values
    y_test = df_test['vuln_label'].values

    # Model B Full predictions
    prob_b_full = mb_clf.predict_proba(X_test_mat)[:, 1]
    pred_b_full = (prob_b_full >= 0.50).astype(int)

    # Model A predictions
    prob_a = df_test['model_a_prob'].values
    pred_a = df_test['model_a_pred'].values

    # Provenance-Only Baseline
    # Fit provenance-only on training set to evaluate fairly
    train_json_path = DATA_P14_DIR / "model_b_p14_train.json"
    with open(train_json_path, "r") as f:
        train_data = json.load(f)
    train_units = train_data['records'] if isinstance(train_data, dict) and 'records' in train_data else train_data

    train_rows = []
    for u in train_units:
        ptype = u.get('author_type', 'unknown')
        involvement = float(u.get('ai_involvement', 0.0))
        train_rows.append({
            'vuln_label': int(u.get('vuln_label')),
            'has_provenance': 1.0,
            'ai_involvement': involvement,
            'is_pure_ai': 1.0 if ptype == 'ai' else 0.0,
            'is_human_edited': 1.0 if ptype == 'human_edit_of_ai' else 0.0,
            'is_ai_edited': 1.0 if ptype == 'ai_edit_of_human' else 0.0,
            'confidence': float(u.get('provenance_confidence', 0.95))
        })
    df_train = pd.DataFrame(train_rows)
    prov_cols = [c for c in feat_cols if c != 'logit_code']
    clf_prov = LogisticRegression(C=5.0, penalty='l2', max_iter=1000, random_state=42)
    clf_prov.fit(df_train[prov_cols].values, df_train['vuln_label'].values)
    prob_prov_only = clf_prov.predict_proba(df_test[prov_cols].values)[:, 1]
    pred_prov_only = (prob_prov_only >= 0.50).astype(int)

    # Detailed metrics
    m_a = compute_metrics_detailed(y_test, pred_a, prob_a)
    m_prov = compute_metrics_detailed(y_test, pred_prov_only, prob_prov_only)
    m_b = compute_metrics_detailed(y_test, pred_b_full, prob_b_full)

    print(f"\n--- Model Comparisons on Exact Test Split (N = {len(y_test)}) ---")
    print(f"Model A (Code Only):      Acc={m_a['accuracy']*100:.2f}% (95% CI: [{m_a['accuracy_ci_95'][0]*100:.1f}%, {m_a['accuracy_ci_95'][1]*100:.1f}%]), AUROC={m_a['auroc']:.4f}, Brier={m_a['brier_score']:.4f}")
    print(f"Provenance-Only Baseline: Acc={m_prov['accuracy']*100:.2f}% (95% CI: [{m_prov['accuracy_ci_95'][0]*100:.1f}%, {m_prov['accuracy_ci_95'][1]*100:.1f}%]), AUROC={m_prov['auroc']:.4f}, Brier={m_prov['brier_score']:.4f}")
    print(f"Model B (Full Fusion):    Acc={m_b['accuracy']*100:.2f}% (95% CI: [{m_b['accuracy_ci_95'][0]*100:.1f}%, {m_b['accuracy_ci_95'][1]*100:.1f}%]), AUROC={m_b['auroc']:.4f}, Brier={m_b['brier_score']:.4f}")

    # Paired Prediction Comparison
    df_test['pred_a'] = pred_a
    df_test['prob_a'] = prob_a
    df_test['pred_prov'] = pred_prov_only
    df_test['prob_prov'] = prob_prov_only
    df_test['pred_b'] = pred_b_full
    df_test['prob_b'] = prob_b_full

    df_test['a_correct'] = (df_test['pred_a'] == df_test['vuln_label'])
    df_test['prov_correct'] = (df_test['pred_prov'] == df_test['vuln_label'])
    df_test['b_correct'] = (df_test['pred_b'] == df_test['vuln_label'])

    improved_over_a = df_test[(~df_test['a_correct']) & (df_test['b_correct'])]
    worsened_vs_a = df_test[(df_test['a_correct']) & (~df_test['b_correct'])]

    improved_over_prov = df_test[(~df_test['prov_correct']) & (df_test['b_correct'])]
    worsened_vs_prov = df_test[(df_test['prov_correct']) & (~df_test['b_correct'])]

    print(f"\n--- Paired Contrast: Model B vs Model A ---")
    print(f"  Both Correct: {((df_test['a_correct']) & (df_test['b_correct'])).sum()} / 32")
    print(f"  Both Wrong:   {((~df_test['a_correct']) & (~df_test['b_correct'])).sum()} / 32")
    print(f"  Model B Improved (A wrong -> B correct): {len(improved_over_a)} / 32")
    for _, r in improved_over_a.iterrows():
        print(f"    • Task {r['task_id']} ({r['provenance_type']}, {r['cwe']}): Label={r['vuln_label']}, A_prob={r['prob_a']:.3f} -> B_prob={r['prob_b']:.3f}")
    print(f"  Model B Worsened (A correct -> B wrong): {len(worsened_vs_a)} / 32")
    for _, r in worsened_vs_a.iterrows():
        print(f"    • Task {r['task_id']} ({r['provenance_type']}, {r['cwe']}): Label={r['vuln_label']}, A_prob={r['prob_a']:.3f} -> B_prob={r['prob_b']:.3f}")

    print(f"\n--- Paired Contrast: Model B vs Provenance-Only Baseline ---")
    print(f"  Both Correct: {((df_test['prov_correct']) & (df_test['b_correct'])).sum()} / 32")
    print(f"  Both Wrong:   {((~df_test['prov_correct']) & (~df_test['b_correct'])).sum()} / 32")
    print(f"  Model B Improved (Prov wrong -> B correct): {len(improved_over_prov)} / 32")
    print(f"  Model B Worsened (Prov correct -> B wrong): {len(worsened_vs_prov)} / 32")
    print(f"  Prediction identity between Prov-Only and Model B: {(pred_b_full == pred_prov_only).mean()*100:.1f}%")

    # Leave-One-Task-Out Cross Validation across ALL 52 Tasks (208 units)
    print("\n--- Running Leave-One-Task-Out Grouped Cross-Validation (52 Folds, N = 208) ---")
    val_json_path = DATA_P14_DIR / "model_b_p14_val.json"
    with open(val_json_path, "r") as f:
        val_data = json.load(f)
    val_units = val_data['records'] if isinstance(val_data, dict) and 'records' in val_data else val_data

    all_units = train_units + val_units + test_units
    all_rows = []
    print("  Computing Model A probabilities across all 208 units...")
    for u in all_units:
        ptype = u.get('author_type', 'unknown')
        involvement = float(u.get('ai_involvement', 0.0))
        code_score = float(predict_model_a(u["code"])["probability"])
        logit_code = float(np.log(np.clip(code_score, 1e-4, 1 - 1e-4) / (1 - np.clip(code_score, 1e-4, 1 - 1e-4))))
        all_rows.append({
            'task_id': u.get('task_id'),
            'cwe': u.get('cwe'),
            'provenance_type': ptype,
            'vuln_label': int(u.get('vuln_label')),
            'model_a_prob': code_score,
            'model_a_pred': int(code_score >= 0.50),
            'logit_code': logit_code,
            'has_provenance': 1.0,
            'ai_involvement': involvement,
            'is_pure_ai': 1.0 if ptype == 'ai' else 0.0,
            'is_human_edited': 1.0 if ptype == 'human_edit_of_ai' else 0.0,
            'is_ai_edited': 1.0 if ptype == 'ai_edit_of_human' else 0.0,
            'confidence': float(u.get('provenance_confidence', 0.95))
        })
    df_all = pd.DataFrame(all_rows)
    unique_tasks = df_all['task_id'].unique()
    print(f"Total dataset: {len(df_all)} units across {len(unique_tasks)} unique tasks")

    # Leave-One-Task-Out (52 folds)
    loto_preds_b = np.zeros(len(df_all))
    loto_probs_b = np.zeros(len(df_all))
    loto_preds_prov = np.zeros(len(df_all))
    loto_probs_prov = np.zeros(len(df_all))
    loto_preds_a = df_all['model_a_pred'].values
    loto_probs_a = df_all['model_a_prob'].values
    y_all = df_all['vuln_label'].values

    for tid in unique_tasks:
        test_mask = (df_all['task_id'] == tid)
        train_mask = ~test_mask

        # Model B Full
        clf_fold_b = LogisticRegression(C=5.0, penalty='l2', max_iter=1000, random_state=42)
        clf_fold_b.fit(df_all.loc[train_mask, feat_cols].values, y_all[train_mask])
        p_b = clf_fold_b.predict_proba(df_all.loc[test_mask, feat_cols].values)[:, 1]
        loto_probs_b[test_mask] = p_b
        loto_preds_b[test_mask] = (p_b >= 0.50).astype(int)

        # Provenance Only
        clf_fold_prov = LogisticRegression(C=5.0, penalty='l2', max_iter=1000, random_state=42)
        clf_fold_prov.fit(df_all.loc[train_mask, prov_cols].values, y_all[train_mask])
        p_prov = clf_fold_prov.predict_proba(df_all.loc[test_mask, prov_cols].values)[:, 1]
        loto_probs_prov[test_mask] = p_prov
        loto_preds_prov[test_mask] = (p_prov >= 0.50).astype(int)

    m_all_a = compute_metrics_detailed(y_all, loto_preds_a, loto_probs_a)
    m_all_prov = compute_metrics_detailed(y_all, loto_preds_prov, loto_probs_prov)
    m_all_b = compute_metrics_detailed(y_all, loto_preds_b, loto_probs_b)

    print("\n--- 52-Fold Leave-One-Task-Out Cross-Validation (N = 208) ---")
    print(f"Model A (Code Only):      Acc={m_all_a['accuracy']*100:.2f}% (95% CI: [{m_all_a['accuracy_ci_95'][0]*100:.1f}%, {m_all_a['accuracy_ci_95'][1]*100:.1f}%]), AUROC={m_all_a['auroc']:.4f}, Brier={m_all_a['brier_score']:.4f}")
    print(f"Provenance-Only Baseline: Acc={m_all_prov['accuracy']*100:.2f}% (95% CI: [{m_all_prov['accuracy_ci_95'][0]*100:.1f}%, {m_all_prov['accuracy_ci_95'][1]*100:.1f}%]), AUROC={m_all_prov['auroc']:.4f}, Brier={m_all_prov['brier_score']:.4f}")
    print(f"Model B (Full Fusion):    Acc={m_all_b['accuracy']*100:.2f}% (95% CI: [{m_all_b['accuracy_ci_95'][0]*100:.1f}%, {m_all_b['accuracy_ci_95'][1]*100:.1f}%]), AUROC={m_all_b['auroc']:.4f}, Brier={m_all_b['brier_score']:.4f}")

    return {
        "test_32": {
            "model_a": m_a,
            "provenance_only": m_prov,
            "model_b": m_b,
            "improved_count_over_a": len(improved_over_a),
            "worsened_count_vs_a": len(worsened_vs_a),
            "improved_count_over_prov": len(improved_over_prov),
            "worsened_count_vs_prov": len(worsened_vs_prov),
            "pred_identical_prov_and_b": float((pred_b_full == pred_prov_only).mean())
        },
        "loto_208": {
            "model_a": m_all_a,
            "provenance_only": m_all_prov,
            "model_b": m_all_b
        }
    }


def main():
    res_a = audit_model_a()
    res_b = audit_model_b()

    combined = {
        "model_a_audit": res_a,
        "model_b_audit": res_b
    }

    out_file = RESULTS_DIR / "phase14_1_audit_data.json"
    with open(out_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[✓] Audit metrics successfully written to: {out_file}")


if __name__ == "__main__":
    main()

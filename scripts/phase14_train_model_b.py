#!/usr/bin/env python3
"""
Phase 14 Model B Training, Validation Tuning & Comprehensive Scientific Ablations
================================================================================
Trains and evaluates Phase 14 Model B (Provenance-Aware Risk Assessor) on:
- Phase 14 Train Split (data/phase14_model_b/model_b_p14_train.json, N=144, 36 tasks)
- Phase 14 Validation Split (model_b_p14_val.json, N=32, 8 tasks) for hyperparameter & architecture tuning
- Phase 14 Locked Test Split (model_b_p14_test.json, N=32, 8 tasks) for frozen independent evaluation
- Historical Locked Benchmark 32 (data/locked_benchmark_32/controlled_benchmark_32.json, N=32)

Executes complete 7-regime ablation suite:
1. Majority Class Baseline
2. Provenance-Only Baseline
3. Model A Code-Only Baseline
4. Model B Full Fusion (Code + Provenance + Interaction)
5. Model B Unknown Provenance Fallback
6. Model B Masked Features
7. Model B Permuted Provenance (Shuffled)
+ Subgroup evaluation across all 4 provenance categories
+ Operational review triage priority distribution
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import predict_model_a

DATA_DIR = PROJECT_ROOT / "data" / "phase14_model_b"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase14"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_fusion_features(
    record: Dict[str, Any],
    p_code: float,
    has_provenance: bool = True,
    include_interactions: bool = True
) -> np.ndarray:
    """
    Constructs feature vector for Model B.
    Base Schema: [logit_code, has_provenance, ai_involvement, is_pure_ai, is_human_edited, is_ai_edited, confidence]
    Extended Schema (+2 interactions): [..., logit_code * ai_involvement, logit_code * is_human_edited]
    """
    eps = 1e-4
    logit_code = np.log((p_code + eps) / (1.0 - p_code + eps))

    if not has_provenance:
        if include_interactions:
            return np.array([logit_code, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        return np.array([logit_code, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    author_type = record.get("author_type", "unknown")
    ai_involvement = float(record.get("ai_involvement", 0.0))
    confidence = float(record.get("provenance_confidence", 1.0))

    is_pure_ai = 1.0 if author_type == "ai" else 0.0
    is_human_edited = 1.0 if author_type == "human_edit_of_ai" else 0.0
    is_ai_edited = 1.0 if author_type == "ai_edit_of_human" else 0.0

    base_feat = [
        logit_code,
        1.0,
        ai_involvement,
        is_pure_ai,
        is_human_edited,
        is_ai_edited,
        confidence
    ]

    if include_interactions:
        inter_ai = logit_code * ai_involvement
        inter_human = logit_code * is_human_edited
        base_feat.extend([inter_ai, inter_human])

    return np.array(base_feat, dtype=float)


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


def load_split(split_name: str):
    path = DATA_DIR / f"model_b_p14_{split_name}.json"
    with open(path, "r") as f:
        return json.load(f)["records"]


def main():
    print("=" * 80)
    print("PHASE 14 — STEP 5: MODEL B ARCHITECTURE, TRAINING & ABLATIONS")
    print("=" * 80)

    # 1. Load Partitions
    train_recs = load_split("train")
    val_recs = load_split("val")
    test_recs = load_split("test")
    print(f"Loaded Phase 14 Partitions: Train={len(train_recs)} (36 tasks), Val={len(val_recs)} (8 tasks), Test={len(test_recs)} (8 tasks)")

    # 2. Extract Model A probabilities for all partitions
    print("\n[1] Running Model A inference across all splits...")
    def process_records(records):
        X_base, X_inter, y, p_codes = [], [], [], []
        for r in records:
            pred_a = predict_model_a(r["code"])
            p_c = pred_a["probability"]
            p_codes.append(p_c)
            y.append(int(r["vuln_label"]))
            X_base.append(extract_fusion_features(r, p_c, has_provenance=True, include_interactions=False))
            X_inter.append(extract_fusion_features(r, p_c, has_provenance=True, include_interactions=True))
        return np.array(X_base), np.array(X_inter), np.array(y), np.array(p_codes)

    X_train_base, X_train_inter, y_train, p_code_train = process_records(train_recs)
    X_val_base, X_val_inter, y_val, p_code_val = process_records(val_recs)
    X_test_base, X_test_inter, y_test, p_code_test = process_records(test_recs)

    # 3. Validation Tuning: Compare Base Schema vs Interaction Schema and tune C
    print("\n[2] Validation Tuning: Comparing Base Schema vs Interaction Schema...")
    results_tuning = {}
    
    # 3a. Base 7-feature schema
    best_c_base, best_auc_base = 1.0, 0.0
    for c_cand in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        clf = LogisticRegression(C=c_cand, penalty='l2', max_iter=1000, random_state=42)
        clf.fit(X_train_base, y_train)
        p_val = clf.predict_proba(X_val_base)[:, 1]
        auc = roc_auc_score(y_val, p_val)
        if auc > best_auc_base:
            best_auc_base = auc
            best_c_base = c_cand
    results_tuning["base_schema"] = {"best_C": best_c_base, "val_auroc": best_auc_base}
    print(f"  Base 7-feature schema: Best C = {best_c_base} (Val AUROC = {best_auc_base:.4f})")

    # 3b. Interaction 9-feature schema
    best_c_inter, best_auc_inter = 1.0, 0.0
    for c_cand in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        clf = LogisticRegression(C=c_cand, penalty='l2', max_iter=1000, random_state=42)
        clf.fit(X_train_inter, y_train)
        p_val = clf.predict_proba(X_val_inter)[:, 1]
        auc = roc_auc_score(y_val, p_val)
        if auc > best_auc_inter:
            best_auc_inter = auc
            best_c_inter = c_cand
    results_tuning["interaction_schema"] = {"best_C": best_c_inter, "val_auroc": best_auc_inter}
    print(f"  Interaction 9-feature schema: Best C = {best_c_inter} (Val AUROC = {best_auc_inter:.4f})")

    # Select champion schema
    use_interactions = (best_auc_inter >= best_auc_base)
    chosen_c = best_c_inter if use_interactions else best_c_base
    X_train_final = X_train_inter if use_interactions else X_train_base
    X_val_final = X_val_inter if use_interactions else X_val_base
    X_test_final = X_test_inter if use_interactions else X_test_base

    feature_names = [
        "logit_code", "has_provenance", "ai_involvement",
        "is_pure_ai", "is_human_edited", "is_ai_edited", "confidence"
    ]
    if use_interactions:
        feature_names.extend(["interaction_code_ai", "interaction_code_human"])

    # Fit Champion Model B on Train + Val
    X_full_train = np.vstack([X_train_final, X_val_final])
    y_full_train = np.concatenate([y_train, y_val])
    clf_champion = LogisticRegression(C=chosen_c, penalty='l2', max_iter=1000, random_state=42)
    clf_champion.fit(X_full_train, y_full_train)

    # Save Checkpoint
    ckpt_path = MODELS_DIR / "phase14_model_b_fusion_best.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "clf_fusion": clf_champion,
            "feature_names": feature_names,
            "best_c": chosen_c,
            "use_interactions": use_interactions,
            "val_score": best_auc_inter if use_interactions else best_auc_base,
            "weights": {name: round(float(w), 4) for name, w in zip(feature_names, clf_champion.coef_[0])},
            "bias": round(float(clf_champion.intercept_[0]), 4),
            "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f)
    print(f"\n[✓] Saved Champion Model B Checkpoint to: {ckpt_path}")
    print(f"    Learned Weights:")
    for name, w in zip(feature_names, clf_champion.coef_[0]):
        print(f"      • {name:24}: {w:+.4f}")
    print(f"      • Intercept               : {clf_champion.intercept_[0]:+.4f}")

    # =========================================================================
    # 4. FULL SCIENTIFIC ABLATIONS ON HELD-OUT TEST PARTITION (N = 32)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCIENTIFIC ABLATION SUITE ON PHASE 14 HELD-OUT TEST PARTITION (N = 32)")
    print("=" * 80)

    # Predictions under various regimes
    # Regime 1: Majority Class (all 0 or all 1, here balanced so constant 0.50)
    p_majority = np.full(len(y_test), 0.50)
    m_majority = compute_metrics(y_test, np.zeros(len(y_test), dtype=int), p_majority)

    # Regime 2: Provenance-Only Baseline (masks logit_code to 0.0)
    X_prov_only = np.copy(X_test_final)
    X_prov_only[:, 0] = 0.0  # Zero out logit_code
    if use_interactions:
        X_prov_only[:, 7:] = 0.0
    p_prov_only = clf_champion.predict_proba(X_prov_only)[:, 1]
    m_prov_only = compute_metrics(y_test, (p_prov_only >= 0.50).astype(int), p_prov_only)

    # Regime 3: Model A Code-Only Baseline
    m_code_only = compute_metrics(y_test, (p_code_test >= 0.50).astype(int), p_code_test)

    # Regime 4: Model B Full Fusion (Code + Provenance)
    p_full = clf_champion.predict_proba(X_test_final)[:, 1]
    m_full = compute_metrics(y_test, (p_full >= 0.50).astype(int), p_full)

    # Regime 5: Model B Missing Provenance Fallback
    # When provenance is unknown, fallback p = p_code
    p_fallback = np.copy(p_code_test)
    m_fallback = compute_metrics(y_test, (p_fallback >= 0.50).astype(int), p_fallback)

    # Regime 6: Model B Masked Features (author category masked to 0)
    X_masked = np.copy(X_test_final)
    X_masked[:, 3:6] = 0.0  # is_pure_ai, is_human_edited, is_ai_edited zeroed
    p_masked = clf_champion.predict_proba(X_masked)[:, 1]
    m_masked = compute_metrics(y_test, (p_masked >= 0.50).astype(int), p_masked)

    # Regime 7: Model B Permuted Provenance (Provenance features randomly shuffled)
    np.random.seed(42)
    perm_indices = np.random.permutation(len(y_test))
    X_perm = np.copy(X_test_final)
    X_perm[:, 1:] = X_test_final[perm_indices, 1:]  # Keep code risk, shuffle provenance
    p_perm = clf_champion.predict_proba(X_perm)[:, 1]
    m_perm = compute_metrics(y_test, (p_perm >= 0.50).astype(int), p_perm)

    ablations = {
        "Majority-Class Baseline": m_majority,
        "Provenance-Only Baseline": m_prov_only,
        "Model A: Code Only": m_code_only,
        "Model B: Full (Code + Prov)": m_full,
        "Model B: Fallback (Unknown)": m_fallback,
        "Model B: Masked Category": m_masked,
        "Model B: Permuted Prov": m_perm
    }

    print(f"\n{'Ablation Regime':32} | {'Accuracy':8} | {'AUROC':7} | {'MCC':7} | {'F1':6} | {'Brier':7}")
    print("-" * 75)
    for name, m in ablations.items():
        print(f"{name:32} | {m['accuracy']*100:6.2f}% | {m['auroc']:7.4f} | {m['mcc']:+7.4f} | {m['f1']:6.4f} | {m['brier_score']:7.4f}")

    # =========================================================================
    # 5. SUBGROUP PERFORMANCE BY PROVENANCE CATEGORY (N = 32)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUBGROUP PERFORMANCE BY PROVENANCE CATEGORY (HELD-OUT TEST SET, N = 32)")
    print("=" * 80)

    subgroup_results = {}
    authors = [r["author_type"] for r in test_recs]
    for auth in ["human", "ai", "human_edit_of_ai", "ai_edit_of_human"]:
        mask = np.array([a == auth for a in authors])
        y_sub = y_test[mask]
        p_sub_a = p_code_test[mask]
        p_sub_b = p_full[mask]
        m_sub_a = compute_metrics(y_sub, (p_sub_a >= 0.50).astype(int), p_sub_a)
        m_sub_b = compute_metrics(y_sub, (p_sub_b >= 0.50).astype(int), p_sub_b)
        subgroup_results[auth] = {
            "n_samples": int(mask.sum()),
            "vuln_samples": int(y_sub.sum()),
            "model_a": m_sub_a,
            "model_b": m_sub_b
        }
        print(f"Subgroup: {auth:20} (N={mask.sum()}, Vuln={y_sub.sum()}): Model A Acc={m_sub_a['accuracy']*100:.1f}%, Brier={m_sub_a['brier_score']:.4f} | Model B Acc={m_sub_b['accuracy']*100:.1f}%, Brier={m_sub_b['brier_score']:.4f}")

    # =========================================================================
    # 6. EVALUATION ON HISTORICAL LOCKED BENCHMARK 32 (N = 32)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SEPARATE EVALUATION ON HISTORICAL LOCKED BENCHMARK 32 (N = 32)")
    print("=" * 80)

    from src.experiments.controlled_editing import create_multi_cwe_benchmark
    benchmarks = create_multi_cwe_benchmark()
    b32_recs = []
    for exp in benchmarks:
        for v in exp.versions:
            b32_recs.append({
                "task_id": exp.task_id,
                "version_id": v.version_id,
                "code": v.code,
                "language": v.language,
                "author_type": v.author_type,
                "ai_involvement": v.ai_involvement,
                "vuln_label": v.vuln_label,
                "provenance_confidence": 1.0,
                "human_edit_count": 2 if "human" in v.author_type else 0
            })

    X_b32_inter, y_b32, p_code_b32 = [], [], []
    for r in b32_recs:
        pred_a = predict_model_a(r["code"])
        p_c = pred_a["probability"]
        p_code_b32.append(p_c)
        y_b32.append(int(r["vuln_label"]))
        X_b32_inter.append(extract_fusion_features(r, p_c, has_provenance=True, include_interactions=use_interactions))

    X_b32_inter = np.array(X_b32_inter)
    y_b32 = np.array(y_b32)
    p_code_b32 = np.array(p_code_b32)

    p_b32_full = clf_champion.predict_proba(X_b32_inter)[:, 1]
    m_b32_a = compute_metrics(y_b32, (p_code_b32 >= 0.50).astype(int), p_code_b32)
    m_b32_b = compute_metrics(y_b32, (p_b32_full >= 0.50).astype(int), p_b32_full)

    print(f"  • Model A (Code-Only):    Acc={m_b32_a['accuracy']*100:.2f}%, AUROC={m_b32_a['auroc']:.4f}, Brier={m_b32_a['brier_score']:.4f}")
    print(f"  • Model B (P14 Champion): Acc={m_b32_b['accuracy']*100:.2f}%, AUROC={m_b32_b['auroc']:.4f}, Brier={m_b32_b['brier_score']:.4f}")

    # =========================================================================
    # 7. OPERATIONAL REVIEW QUEUE TRIAGE METRICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("OPERATIONAL REVIEW QUEUE TRIAGE POLICY ANALYSIS")
    print("=" * 80)

    # Triage policy: Priority score = P_final * (1 + 0.5 * AI_involvement)
    triage_scores = []
    tiers = []
    for r, p_f in zip(test_recs, p_full):
        ai_inv = float(r.get("ai_involvement", 0.0))
        is_p_ai = (r.get("author_type") == "ai")
        t_score = p_f * (1.0 + 0.5 * ai_inv)
        triage_scores.append(t_score)
        
        if is_p_ai and p_f >= 0.40:
            tier = "Tier-1 (Urgent)"
        elif t_score >= 0.60 or p_f >= 0.60:
            tier = "Tier-1 (Urgent)"
        elif t_score >= 0.35 or p_f >= 0.35:
            tier = "Tier-2 (Elevated)"
        else:
            tier = "Tier-3 (Standard)"
        tiers.append(tier)

    tier_series = pd.Series(tiers)
    tier_dist = tier_series.value_counts().to_dict()
    print("Review Triage Distribution across Test Samples:")
    for tier_name, count in tier_dist.items():
        pct = count / len(tiers) * 100
        print(f"  • {tier_name:20}: {count:2} ({pct:.1f}%)")

    # Save comprehensive evaluation json
    eval_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_tasks": 36,
        "training_units": 144,
        "validation_tasks": 8,
        "validation_units": 32,
        "test_tasks": 8,
        "test_units": 32,
        "tuning_results": results_tuning,
        "champion_weights": {name: round(float(w), 4) for name, w in zip(feature_names, clf_champion.coef_[0])},
        "intercept": round(float(clf_champion.intercept_[0]), 4),
        "held_out_test_ablations": ablations,
        "subgroup_results": subgroup_results,
        "historical_benchmark_32": {"model_a": m_b32_a, "model_b": m_b32_b},
        "triage_distribution": tier_dist
    }

    out_file = RESULTS_DIR / "phase14_model_b_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\n[✓] Phase 14 Model B evaluation successfully saved to: {out_file}")


if __name__ == "__main__":
    main()

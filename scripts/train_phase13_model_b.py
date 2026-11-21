#!/usr/bin/env python3
"""
Phase 13 Model B Training, Validation, and Independent Test Evaluation
=====================================================================
DTU B.Tech Project: Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code
Authors: Mankabir Singh (23/IT/199) and Mohnish Gautam (23/IT/200)

This script trains a balanced provenance-aware fusion model on data/phase13_model_b/,
tunes regularization on the validation set, and performs locked evaluation and
full ablation studies on the independent test set.
"""

import sys
import os
import json
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
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

from src.models.inference import predict_model_a

DATA_DIR = PROJECT_ROOT / "data" / "phase13_model_b"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_features(
    record: Dict[str, Any],
    p_code: float,
    has_provenance: bool = True
) -> np.ndarray:
    """
    Constructs feature vector for Model B:
    Schema: [logit_code, has_provenance, ai_involvement, is_pure_ai, is_human_edited, is_ai_edited, confidence]
    """
    eps = 1e-4
    logit_code = np.log((p_code + eps) / (1.0 - p_code + eps))
    
    if not has_provenance:
        return np.array([logit_code, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        
    author_type = record.get("author_type", "unknown")
    ai_involvement = float(record.get("ai_involvement", 0.0))
    confidence = float(record.get("provenance_confidence", 1.0))
    
    is_pure_ai = 1.0 if author_type == "ai" else 0.0
    is_human_edited = 1.0 if author_type == "human_edit_of_ai" else 0.0
    is_ai_edited = 1.0 if author_type == "ai_edit_of_human" else 0.0
    
    return np.array([
        logit_code,
        1.0,  # has_provenance flag
        ai_involvement,
        is_pure_ai,
        is_human_edited,
        is_ai_edited,
        confidence
    ], dtype=float)


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score interval for binomial proportions."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95%
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n) / denom
    return float(max(0.0, center - spread)), float(min(1.0, center + spread))


def compute_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive metric suite."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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
        
    brier = float(brier_score_loss(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    ci_low, ci_high = compute_wilson_ci(int(np.sum(y_true == y_pred)), len(y_true))
    
    return {
        "accuracy": acc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "auroc": auroc,
        "prauc": prauc,
        "brier_score": brier,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "fpr": fpr,
        "fnr": fnr,
        "n_samples": len(y_true)
    }


def run_training_and_evaluation():
    print("=" * 80)
    print("PHASE 13 MODEL B TRAINING & SCIENTIFIC ABLATION SUITE")
    print("=" * 80)
    
    # 1. Load Data
    train_path = DATA_DIR / "model_b_p13_train.json"
    val_path = DATA_DIR / "model_b_p13_val.json"
    test_path = DATA_DIR / "model_b_p13_test_locked.json"
    
    with open(train_path, "r") as f:
        train_records = json.load(f)["records"]
    with open(val_path, "r") as f:
        val_records = json.load(f)["records"]
    with open(test_path, "r") as f:
        test_records = json.load(f)["records"]
        
    print(f"Loaded Partitions: Train={len(train_records)} units (20 tasks), Val={len(val_records)} units (4 tasks), Test={len(test_records)} units (4 tasks)")
    
    # 2. Extract Base Model A Code-Only Probabilities
    print("\nExtracting Model A code-only base probabilities...")
    def process_records(records):
        X_full, X_prov_only, y, p_codes = [], [], [], []
        for r in records:
            res_a = predict_model_a(r["code"], language=r.get("language", "c"))
            p_code = res_a["probability"]
            p_codes.append(p_code)
            y.append(r["vuln_label"])
            
            # Full features [logit_code, has_prov, ai_inv, is_pure_ai, is_human_edited, is_ai_edited, conf]
            feat_full = extract_features(r, p_code, has_provenance=True)
            X_full.append(feat_full)
            
            # Provenance-only features [ai_inv, is_pure_ai, is_human_edited, is_ai_edited, conf]
            X_prov_only.append(feat_full[2:])
            
        return np.array(X_full), np.array(X_prov_only), np.array(y), np.array(p_codes)

    X_train, X_train_po, y_train, p_codes_train = process_records(train_records)
    X_val, X_val_po, y_val, p_codes_val = process_records(val_records)
    X_test, X_test_po, y_test, p_codes_test = process_records(test_records)
    
    feature_names = ["logit_code", "has_provenance", "ai_involvement", "is_pure_ai", "is_human_edited", "is_ai_edited", "confidence"]
    
    # 3. Validation Model Selection
    print("\n--- Model Tuning on Validation Partition (N=16) ---")
    best_c = 1.0
    best_val_score = -1.0
    best_clf = None
    val_tuning_logs = []
    
    for c in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        clf = LogisticRegression(C=c, penalty='l2', solver='lbfgs', random_state=42)
        clf.fit(X_train, y_train)
        probs_val = clf.predict_proba(X_val)[:, 1]
        preds_val = (probs_val >= 0.50).astype(int)
        acc_val = accuracy_score(y_val, preds_val)
        mcc_val = matthews_corrcoef(y_val, preds_val) if len(np.unique(preds_val)) > 1 else 0.0
        
        log_entry = {
            "C": c,
            "val_accuracy": round(float(acc_val), 4),
            "val_mcc": round(float(mcc_val), 4),
            "logit_coef": round(float(clf.coef_[0, 0]), 4)
        }
        val_tuning_logs.append(log_entry)
        print(f"  C={c:5.2f} -> Val Acc={acc_val*100:5.2f}%, Val MCC={mcc_val:+6.4f}, Logit Coef={clf.coef_[0, 0]:.4f}")
        
        if acc_val > best_val_score or (acc_val == best_val_score and clf.coef_[0, 0] > (best_clf.coef_[0, 0] if best_clf else -999)):
            best_val_score = acc_val
            best_c = c
            best_clf = clf
            
    print(f"\nWinning Hyperparameter Selected on Validation: C={best_c} (Val Accuracy = {best_val_score*100:.2f}%)")
    weights_dict = dict(zip(feature_names, [round(float(w), 4) for w in best_clf.coef_[0]]))
    print(f"Learned Feature Weights: {weights_dict}")
    print(f"Learned Bias: {best_clf.intercept_[0]:.4f}")
    
    # Train Provenance-Only Model for Baseline Comparison
    clf_prov_only = LogisticRegression(C=1.0, solver='lbfgs', random_state=42)
    clf_prov_only.fit(X_train_po, y_train)
    
    # 4. Independent Locked Test Set Evaluation (N=16)
    print("\n" + "=" * 80)
    print("PHASE 13 INDEPENDENT LOCKED TEST EVALUATION (N=16) & ABLATION STUDY")
    print("=" * 80)
    
    # Regime 0: Majority Class Baseline
    maj_pred = np.zeros_like(y_test)
    maj_prob = np.full_like(y_test, fill_value=0.5, dtype=float)
    m_maj = compute_metrics(y_test, maj_pred, maj_prob)
    
    # Regime 1: Model A (Code Only)
    preds_a = (p_codes_test >= 0.50).astype(int)
    m_code_only = compute_metrics(y_test, preds_a, p_codes_test)
    
    # Regime 2: Provenance-Only Baseline
    probs_po = clf_prov_only.predict_proba(X_test_po)[:, 1]
    preds_po = (probs_po >= 0.50).astype(int)
    m_prov_only = compute_metrics(y_test, preds_po, probs_po)
    
    # Regime 3: Model B Full (Code + Provenance)
    probs_b = best_clf.predict_proba(X_test)[:, 1]
    preds_b = (probs_b >= 0.50).astype(int)
    m_b_full = compute_metrics(y_test, preds_b, probs_b)
    
    # Regime 4: Model B Fallback (Unknown Provenance)
    m_b_fallback = compute_metrics(y_test, preds_a, p_codes_test)
    
    # Regime 5: Model B Masked (Provenance Features Set to Zero)
    X_test_masked = np.copy(X_test)
    X_test_masked[:, 2:] = 0.0
    probs_masked = best_clf.predict_proba(X_test_masked)[:, 1]
    preds_masked = (probs_masked >= 0.50).astype(int)
    m_b_masked = compute_metrics(y_test, preds_masked, probs_masked)
    
    # Regime 6: Provenance Permutation Ablation Test
    rng = np.random.RandomState(42)
    perm_indices = rng.permutation(len(y_test))
    X_test_perm = np.copy(X_test)
    X_test_perm[:, 2:] = X_test[perm_indices, 2:]  # Permute provenance
    probs_perm = best_clf.predict_proba(X_test_perm)[:, 1]
    preds_perm = (probs_perm >= 0.50).astype(int)
    m_b_perm = compute_metrics(y_test, preds_perm, probs_perm)
    
    ablation_results = [
        ("Majority-Class Baseline", m_maj),
        ("Provenance-Only Baseline", m_prov_only),
        ("Model A: Code Only", m_code_only),
        ("Model B: Full (Code + Provenance)", m_b_full),
        ("Model B: Unknown Provenance Fallback", m_b_fallback),
        ("Model B: Masked Provenance Features", m_b_masked),
        ("Model B: Permuted Provenance Ablation", m_b_perm)
    ]
    
    print(f"\n{'Evaluation Regime':<38} | {'Acc':<7} | {'BalAcc':<7} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'MCC':<8} | {'AUROC':<7} | {'Brier':<7}")
    print("-" * 115)
    for name, m in ablation_results:
        print(f"{name:<38} | {m['accuracy']*100:5.2f}% | {m['balanced_accuracy']*100:5.2f}% | {m['precision']*100:5.2f}% | {m['recall']*100:5.2f}% | {m['f1']:5.4f} | {m['mcc']:+7.4f} | {m['auroc']:5.4f} | {m['brier_score']:5.4f}")
    print("-" * 115)
    
    # 5. Subgroup Breakdown across the 4 Provenance Categories
    print("\n--- Subgroup Breakdown on Locked Test Set (N=16) ---")
    subgroups = {
        "Pure Human Baseline": [i for i, r in enumerate(test_records) if r["author_type"] == "human"],
        "Pure AI Generation": [i for i, r in enumerate(test_records) if r["author_type"] == "ai"],
        "Human-Edited AI": [i for i, r in enumerate(test_records) if r["author_type"] == "human_edit_of_ai"],
        "AI-Edited Human": [i for i, r in enumerate(test_records) if r["author_type"] == "ai_edit_of_human"],
    }
    
    subgroup_metrics = {}
    for sg_name, indices in subgroups.items():
        y_sg = y_test[indices]
        pa_sg = p_codes_test[indices]
        pb_sg = probs_b[indices]
        pred_a_sg = (pa_sg >= 0.50).astype(int)
        pred_b_sg = (pb_sg >= 0.50).astype(int)
        
        acc_a = float(accuracy_score(y_sg, pred_a_sg))
        acc_b = float(accuracy_score(y_sg, pred_b_sg))
        brier_a = float(brier_score_loss(y_sg, pa_sg))
        brier_b = float(brier_score_loss(y_sg, pb_sg))
        
        subgroup_metrics[sg_name] = {
            "n_samples": len(indices),
            "vuln_count": int(np.sum(y_sg == 1)),
            "safe_count": int(np.sum(y_sg == 0)),
            "model_a_acc": acc_a,
            "model_b_acc": acc_b,
            "model_a_brier": brier_a,
            "model_b_brier": brier_b,
            "brier_delta": round(brier_b - brier_a, 4)
        }
        print(f"  {sg_name:<24} (N={len(indices)}): Model A Acc={acc_a*100:5.2f}%, Model B Acc={acc_b*100:5.2f}% | Brier A={brier_a:5.4f}, Brier B={brier_b:5.4f}")
        
    # 6. Evaluation on Prior Locked Benchmark 32
    print("\n--- Separate Evaluation on Prior Locked Benchmark (N=32) ---")
    from src.experiments.controlled_editing import create_multi_cwe_benchmark
    benchmarks = create_multi_cwe_benchmark()
    
    b32_X, b32_y, b32_pa = [], [], []
    for exp in benchmarks:
        for v in exp.versions:
            res_a = predict_model_a(v.code, language=v.language)
            p_c = res_a["probability"]
            b32_pa.append(p_c)
            b32_y.append(v.vuln_label)
            feat = extract_features({
                "author_type": v.author_type,
                "ai_involvement": v.ai_involvement,
                "provenance_confidence": 1.0
            }, p_c, has_provenance=True)
            b32_X.append(feat)
        
    b32_X = np.array(b32_X)
    b32_y = np.array(b32_y)
    b32_pa = np.array(b32_pa)
    
    b32_pb = best_clf.predict_proba(b32_X)[:, 1]
    b32_pred_a = (b32_pa >= 0.50).astype(int)
    b32_pred_b = (b32_pb >= 0.50).astype(int)
    
    m_b32_a = compute_metrics(b32_y, b32_pred_a, b32_pa)
    m_b32_b = compute_metrics(b32_y, b32_pred_b, b32_pb)
    
    print(f"  Prior Benchmark 32 Model A Acc: {m_b32_a['accuracy']*100:.2f}%, AUROC: {m_b32_a['auroc']:.4f}, Brier: {m_b32_a['brier_score']:.4f}")
    print(f"  Prior Benchmark 32 Model B Acc: {m_b32_b['accuracy']*100:.2f}%, AUROC: {m_b32_b['auroc']:.4f}, Brier: {m_b32_b['brier_score']:.4f}")
    
    # 7. Save Checkpoint
    ckpt_path = MODELS_DIR / "phase13_model_b_fusion_best.pkl"
    checkpoint_data = {
        "clf_fusion": best_clf,
        "feature_names": feature_names,
        "best_c": best_c,
        "val_score": best_val_score,
        "weights": weights_dict,
        "bias": float(best_clf.intercept_[0]),
        "test_metrics": m_b_full,
        "subgroup_metrics": subgroup_metrics,
        "prior_benchmark_metrics": m_b32_b,
        "val_tuning_logs": val_tuning_logs,
        "trained_date": "2026-10-27"
    }
    with open(ckpt_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
        
    chk_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    print(f"\nSaved winning Phase 13 checkpoint to {ckpt_path}")
    print(f"Checkpoint SHA-256: {chk_sha}")
    
    # Export full JSON report
    report_json_path = REPORTS_DIR / "phase13_model_b_evaluation.json"
    full_report = {
        "experiment": "Phase 13 Model B Training & Evaluation",
        "dataset_path": str(DATA_DIR),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": chk_sha,
        "model_architecture": "Late Fusion Calibrated Logistic Regression",
        "hyperparameter_c": best_c,
        "learned_weights": weights_dict,
        "learned_bias": float(best_clf.intercept_[0]),
        "ablations": {
            "majority_baseline": m_maj,
            "provenance_only": m_prov_only,
            "model_a_code_only": m_code_only,
            "model_b_full": m_b_full,
            "model_b_fallback": m_b_fallback,
            "model_b_masked": m_b_masked,
            "model_b_permuted": m_b_perm
        },
        "subgroup_metrics": subgroup_metrics,
        "prior_locked_benchmark_32": {
            "model_a": m_b32_a,
            "model_b": m_b32_b
        }
    }
    with open(report_json_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"Exported evaluation JSON to {report_json_path}")
    return full_report


if __name__ == "__main__":
    run_training_and_evaluation()

#!/usr/bin/env python3
"""
Phase 11 Model B Training, Validation, and Independent Test Evaluation
=====================================================================
DTU B.Tech Project: Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code
Authors: Mankabir Singh (23/IT/199) and Mohnish Gautam (23/IT/200)

This script trains a balanced provenance-aware fusion model on data/phase11_model_b/,
tunes regularization on the validation set, and performs locked evaluation and
ablation studies on the independent test set.
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import predict_model_a

DATA_DIR = PROJECT_ROOT / "data" / "phase11_model_b"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
        
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": auroc,
        "pr_auc": prauc,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "fpr": fpr,
        "fnr": fnr
    }


def main():
    print("=" * 80)
    print("PHASE 11 MODEL B: TRAINING, VALIDATION, AND ABLATION EXPERIMENTS")
    print("=" * 80)
    
    # 1. Load balanced partitions
    train_path = DATA_DIR / "model_b_p11_train.json"
    val_path = DATA_DIR / "model_b_p11_val.json"
    test_path = DATA_DIR / "model_b_p11_test_locked.json"
    
    with open(train_path, "r") as f:
        train_records = json.load(f)["records"]
    with open(val_path, "r") as f:
        val_records = json.load(f)["records"]
    with open(test_path, "r") as f:
        test_records = json.load(f)["records"]
        
    print(f"Loaded Partitions: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    
    # 2. Extract Model A base probabilities
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
    
    # 3. Model Tuning on Validation Set
    print("\n--- Model Tuning on Validation Partition ---")
    best_c = 1.0
    best_val_score = -1.0
    best_clf = None
    
    for c in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        clf = LogisticRegression(C=c, penalty='l2', solver='lbfgs', random_state=42)
        clf.fit(X_train, y_train)
        probs_val = clf.predict_proba(X_val)[:, 1]
        preds_val = (probs_val >= 0.50).astype(int)
        acc_val = accuracy_score(y_val, preds_val)
        mcc_val = matthews_corrcoef(y_val, preds_val)
        
        print(f"  C={c:5.2f} -> Val Acc={acc_val*100:.2f}%, Val MCC={mcc_val:+.4f}, Logit Coef={clf.coef_[0, 0]:.4f}")
        if acc_val > best_val_score or (acc_val == best_val_score and clf.coef_[0, 0] > best_clf.coef_[0, 0]):
            best_val_score = acc_val
            best_c = c
            best_clf = clf
            
    print(f"\nWinning Model Selected on Validation: C={best_c} (Val Accuracy = {best_val_score*100:.2f}%)")
    print(f"Learned Feature Weights: {dict(zip(feature_names, [round(float(w), 4) for w in best_clf.coef_[0]]))}")
    print(f"Learned Bias: {best_clf.intercept_[0]:.4f}")
    
    # Train Provenance-Only Baseline for Comparison
    clf_prov_only = LogisticRegression(C=1.0, solver='lbfgs', random_state=42)
    clf_prov_only.fit(X_train_po, y_train)
    
    # 4. Evaluation on Locked Independent Test Set (N=16)
    print("\n" + "=" * 80)
    print("INDEPENDENT LOCKED TEST EVALUATION (N=16) & ABLATION STUDY")
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
    
    # Regime 4: Model B with Provenance Unknown (Fallback)
    # When provenance is unknown, X has has_provenance=0 and all prov features=0
    # In our production inference, fallback directly sets P_final = P_code
    m_b_fallback = compute_metrics(y_test, preds_a, p_codes_test)
    
    # Regime 5: Model B with Provenance Features Masked (zeros)
    X_test_masked = np.copy(X_test)
    X_test_masked[:, 2:] = 0.0  # Mask all provenance signals
    probs_masked = best_clf.predict_proba(X_test_masked)[:, 1]
    preds_masked = (probs_masked >= 0.50).astype(int)
    m_b_masked = compute_metrics(y_test, preds_masked, probs_masked)
    
    # Summary Table
    ablation_results = [
        ("Majority-Class Baseline", m_maj),
        ("Provenance-Only Baseline", m_prov_only),
        ("Model A: Code Only", m_code_only),
        ("Model B: Full (Code + Provenance)", m_b_full),
        ("Model B: Unknown Provenance Fallback", m_b_fallback),
        ("Model B: Masked Provenance Features", m_b_masked)
    ]
    
    print(f"\n{'Evaluation Regime':<38} | {'Acc':<7} | {'BalAcc':<7} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'MCC':<8} | {'AUROC':<7}")
    print("-" * 105)
    for name, m in ablation_results:
        print(f"{name:<38} | {m['accuracy']*100:5.2f}% | {m['balanced_accuracy']*100:5.2f}% | {m['precision']*100:5.2f}% | {m['recall']*100:5.2f}% | {m['f1']:5.4f} | {m['mcc']:+7.4f} | {m['auroc']:5.4f}")
        
    print("-" * 105)
    
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
        
        acc_a = accuracy_score(y_sg, pred_a_sg)
        acc_b = accuracy_score(y_sg, pred_b_sg)
        
        subgroup_metrics[sg_name] = {
            "n_samples": len(indices),
            "n_vulnerable": int(np.sum(y_sg)),
            "model_a_acc": float(acc_a),
            "model_b_acc": float(acc_b)
        }
        print(f"  {sg_name:<22} (N={len(indices)}, {np.sum(y_sg)}V/{len(indices)-np.sum(y_sg)}S): Model A Acc = {acc_a*100:5.2f}% | Model B Acc = {acc_b*100:5.2f}%")
        
    # 6. Review Prioritization Evaluation
    # Review budget: top 25% (4 items) and top 50% (8 items)
    print("\n--- Review Prioritization Simulation (Code Audit Budget) ---")
    
    # Priority sorting
    # Model A: sort by p_code descending
    # Model B: sort by triage_score = p_final * (1.0 + 0.5 * ai_involvement)
    ai_inv_test = np.array([r.get("ai_involvement", 0.0) for r in test_records])
    triage_scores_b = probs_b * (1.0 + 0.5 * ai_inv_test)
    
    idx_sorted_a = np.argsort(-p_codes_test)
    idx_sorted_b = np.argsort(-triage_scores_b)
    
    total_vulns = int(np.sum(y_test))
    
    for budget_pct, budget_k in [(0.25, 4), (0.50, 8)]:
        audited_a = idx_sorted_a[:budget_k]
        audited_b = idx_sorted_b[:budget_k]
        
        vulns_found_a = int(np.sum(y_test[audited_a]))
        vulns_found_b = int(np.sum(y_test[audited_b]))
        
        fp_a = budget_k - vulns_found_a
        fp_b = budget_k - vulns_found_b
        
        recall_a = vulns_found_a / total_vulns
        recall_b = vulns_found_b / total_vulns
        
        print(f"  Review Budget = {budget_k} files ({int(budget_pct*100)}% of audit capacity):")
        print(f"    Model A: Found {vulns_found_a}/{total_vulns} vulns ({recall_a*100:.1f}% recall) with {fp_a} false positives")
        print(f"    Model B: Found {vulns_found_b}/{total_vulns} vulns ({recall_b*100:.1f}% recall) with {fp_b} false positives")

    # 7. Export locked predictions CSV
    csv_rows = []
    for i, r in enumerate(test_records):
        csv_rows.append({
            "example_id": r["sample_id"],
            "task_id": r["task_id"],
            "cwe": r["cwe"],
            "cwe_name": r["cwe_name"],
            "author_type": r["author_type"],
            "ai_involvement": r.get("ai_involvement", 0.0),
            "ground_truth_vuln": r["vuln_label"],
            "model_a_prob": round(float(p_codes_test[i]), 4),
            "model_a_pred": int(preds_a[i]),
            "model_b_prob": round(float(probs_b[i]), 4),
            "model_b_pred": int(preds_b[i]),
            "model_b_triage_score": round(float(triage_scores_b[i]), 4),
            "model_b_fallback_prob": round(float(p_codes_test[i]), 4),
            "checkpoint_used": "phase11_model_b_fusion_best.pkl"
        })
    df_preds = pd.DataFrame(csv_rows)
    pred_csv_path = RESULTS_DIR / "phase11_locked_test_predictions.csv"
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"\nExported {len(df_preds)} individual predictions to {pred_csv_path}")

    # 8. Save updated Model B checkpoint
    ckpt_save_path = MODELS_DIR / "phase11_model_b_fusion_best.pkl"
    ckpt_obj = {
        "clf_fusion": best_clf,
        "feature_names": feature_names,
        "best_c": best_c,
        "val_score": best_val_score,
        "weights": dict(zip(feature_names, [float(w) for w in best_clf.coef_[0]])),
        "bias": float(best_clf.intercept_[0]),
        "test_metrics": m_b_full,
        "subgroup_metrics": subgroup_metrics,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(ckpt_save_path, "wb") as f:
        pickle.dump(ckpt_obj, f)
        
    with open(ckpt_save_path, "rb") as f:
        ckpt_sha = hashlib.sha256(f.read()).hexdigest()
        
    print(f"Saved winning Model B checkpoint to {ckpt_save_path}")
    print(f"Checkpoint SHA-256: {ckpt_sha}")
    print("\nPhase 11 Model B Training and Evaluation Complete!")


if __name__ == "__main__":
    main()

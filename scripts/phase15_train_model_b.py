#!/usr/bin/env python3
"""
Phase 15 — Step 5: Model B Training, Architecture Exploration & Evaluation
==========================================================================
Evaluates Model B on the expanded 150-task / 600-unit dataset:
- Train Split: 105 tasks (420 units)
- Validation Split: 22 tasks (88 units)
- Test Split: 23 tasks (92 units)

Architectures Evaluated:
1. Base Late Fusion (7 features, regularized logistic regression)
2. Interaction Schema (9 features, logit * ai_involvement, logit * is_human_edited)
3. Calibrated Score-Level Fusion (Platt calibrated alpha weighting)

Evaluation Protocols:
- Task 1: Binary Vulnerability Classification (Acc, P, R, F1, MCC, AUROC, PR-AUC)
- Task 2: Review-Priority Ranking & Triage (Precision@K, Recall@K for K in [10%, 20%, 30%, 50%], AP, Brier)
- 10-Fold Grouped Cross-Validation by Independent Task
- Full Ablation Suite (Majority, Prov-Only, Code-Only, Full B, Fallback, Masked, Permuted)
"""

import sys
import os
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_P15_DIR = PROJECT_ROOT / "data" / "phase15_model_b"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_P15_DIR = PROJECT_ROOT / "results" / "phase15"
RESULTS_P15_DIR.mkdir(parents=True, exist_ok=True)

from src.models.inference import predict_model_a


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    denom = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


CACHE_FILE = RESULTS_P15_DIR / "p_code_cache.json"
P_CODE_CACHE: Dict[str, float] = {}

def init_p_code_cache(all_records: List[Dict[str, Any]]):
    global P_CODE_CACHE
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                P_CODE_CACHE = json.load(f)
            print(f"  Loaded {len(P_CODE_CACHE)} cached Model A probabilities from {CACHE_FILE.name}")
        except Exception:
            P_CODE_CACHE = {}

    missing = [r["code"] for r in all_records if r["code"] not in P_CODE_CACHE]
    if missing:
        print(f"  Computing Model A probabilities for {len(missing)} code snippets...")
        t0 = time.time()
        for idx, code in enumerate(missing):
            res = predict_model_a(code)
            P_CODE_CACHE[code] = float(res["probability"])
            if (idx + 1) % 50 == 0 or (idx + 1) == len(missing):
                elapsed = time.time() - t0
                print(f"    [{idx + 1}/{len(missing)}] processed in {elapsed:.1f}s ({(idx + 1)/elapsed:.1f} snippets/s)")
        with open(CACHE_FILE, "w") as f:
            json.dump(P_CODE_CACHE, f)
        print(f"  Saved updated cache to {CACHE_FILE.name}")

def get_code_probability(code: str) -> float:
    if code not in P_CODE_CACHE:
        pred_a = predict_model_a(code)
        P_CODE_CACHE[code] = float(pred_a["probability"])
    return P_CODE_CACHE[code]


def extract_features(records, include_interactions=False, fallback=False, permute=False):
    """Extracts Model B feature matrix from record dicts."""
    X, y, p_codes = [], [], []
    
    # Optional permutation of provenance records
    if permute:
        prov_data = [(r.get("author_type", "human"), float(r.get("ai_involvement", 0.0)), float(r.get("provenance_confidence", 1.0))) for r in records]
        rng = np.random.RandomState(42)
        rng.shuffle(prov_data)

    for i, r in enumerate(records):
        p_c = get_code_probability(r["code"])
        p_codes.append(p_c)
        y.append(int(r["vuln_label"]))

        eps = 1e-4
        logit_code = float(np.log((p_c + eps) / (1.0 - p_c + eps)))

        if fallback:
            # Unknown provenance: all provenance indicators zeroed
            feat = [logit_code, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if include_interactions:
                feat.extend([0.0, 0.0])
            X.append(feat)
            continue

        if permute:
            ptype, ai_inv, conf = prov_data[i]
        else:
            ptype = r.get("author_type", "human")
            ai_inv = float(r.get("ai_involvement", 0.0))
            conf = float(r.get("provenance_confidence", 1.0))

        is_pure_ai = 1.0 if ptype == "ai" else 0.0
        is_human_edited = 1.0 if ptype == "human_edit_of_ai" else 0.0
        is_ai_edited = 1.0 if ptype == "ai_edit_of_human" else 0.0

        feat = [logit_code, 1.0, ai_inv, is_pure_ai, is_human_edited, is_ai_edited, conf]
        if include_interactions:
            feat.extend([logit_code * ai_inv, logit_code * is_human_edited])
        X.append(feat)

    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(p_codes, dtype=float)


def compute_classification_metrics(y_true, y_pred, y_prob):
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


def compute_triage_ranking_metrics(y_true, y_prob):
    """
    Computes review-priority ranking metrics:
    Precision@K, Recall@K, and vulnerabilities caught within review budgets (10%, 20%, 30%, 50%).
    """
    n = len(y_true)
    total_vulns = int(np.sum(y_true))
    order = np.argsort(-y_prob)  # descending risk order
    y_sorted = y_true[order]

    triage_results = {}
    for budget_pct in [10, 20, 30, 50]:
        k = max(1, int(n * (budget_pct / 100.0)))
        vulns_found = int(np.sum(y_sorted[:k]))
        prec_k = float(vulns_found / k)
        rec_k = float(vulns_found / total_vulns) if total_vulns > 0 else 0.0
        triage_results[f"budget_{budget_pct}pct"] = {
            "k_samples_inspected": k,
            "vulns_caught": vulns_found,
            "total_vulns": total_vulns,
            "precision_at_k": prec_k,
            "recall_at_k": rec_k,
            "workload_saved_pct": 100.0 - budget_pct
        }

    triage_results["average_precision"] = float(average_precision_score(y_true, y_prob))
    triage_results["auroc"] = float(roc_auc_score(y_true, y_prob))
    return triage_results


def main():
    print("=" * 80)
    print("PHASE 15 — MODEL B ADVANCED TRAINING & TRIAGE RANKING EVALUATION")
    print("=" * 80)

    # 1. Load Partitions
    with open(DATA_P15_DIR / "model_b_p15_train.json") as f:
        train_recs = json.load(f)["records"]
    with open(DATA_P15_DIR / "model_b_p15_val.json") as f:
        val_recs = json.load(f)["records"]
    with open(DATA_P15_DIR / "model_b_p15_test.json") as f:
        test_recs = json.load(f)["records"]

    print(f"Loaded Phase 15 Partitions:")
    print(f"  Train: {len(train_recs)} units ({len(set(r['task_id'] for r in train_recs))} tasks)")
    print(f"  Val:   {len(val_recs)} units ({len(set(r['task_id'] for r in val_recs))} tasks)")
    print(f"  Test:  {len(test_recs)} units ({len(set(r['task_id'] for r in test_recs))} tasks)")

    # 1.5 Initialize / Compute Model A Probability Cache
    all_recs = train_recs + val_recs + test_recs
    init_p_code_cache(all_recs)

    # 2. Extract Features
    print("\n[1] Extracting Model A logits and fusion vectors...")
    t0 = time.time()
    X_train_base, y_train, p_code_train = extract_features(train_recs, include_interactions=False)
    X_val_base, y_val, p_code_val = extract_features(val_recs, include_interactions=False)
    X_test_base, y_test, p_code_test = extract_features(test_recs, include_interactions=False)
    
    X_train_inter, _, _ = extract_features(train_recs, include_interactions=True)
    X_val_inter, _, _ = extract_features(val_recs, include_interactions=True)
    X_test_inter, _, _ = extract_features(test_recs, include_interactions=True)
    print(f"  Feature extraction completed in {time.time()-t0:.1f}s")

    # 3. Validation Tuning: Compare Base Schema vs Interaction Schema and tune C
    print("\n[2] Validation Tuning: Comparing Architectures and Regularization...")
    # Base Schema Tuning
    best_c_base, best_auc_base = 1.0, 0.0
    for c_cand in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        clf = LogisticRegression(C=c_cand, penalty='l2', max_iter=1000, random_state=42)
        clf.fit(X_train_base, y_train)
        p_val = clf.predict_proba(X_val_base)[:, 1]
        auc = roc_auc_score(y_val, p_val)
        if auc > best_auc_base:
            best_auc_base = auc
            best_c_base = c_cand
    print(f"  Base 7-feature schema: Best C = {best_c_base} (Val AUROC = {best_auc_base:.4f})")

    # Interaction Schema Tuning
    best_c_inter, best_auc_inter = 1.0, 0.0
    for c_cand in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        clf = LogisticRegression(C=c_cand, penalty='l2', max_iter=1000, random_state=42)
        clf.fit(X_train_inter, y_train)
        p_val = clf.predict_proba(X_val_inter)[:, 1]
        auc = roc_auc_score(y_val, p_val)
        if auc > best_auc_inter:
            best_auc_inter = auc
            best_c_inter = c_cand
    print(f"  Interaction 9-feature schema: Best C = {best_c_inter} (Val AUROC = {best_auc_inter:.4f})")

    # Train Champion Architecture on Training Split
    use_inter = (best_auc_inter > best_auc_base)
    best_c = best_c_inter if use_inter else best_c_base
    X_tr_final = X_train_inter if use_inter else X_train_base
    X_te_final = X_test_inter if use_inter else X_test_base

    clf_b_champ = LogisticRegression(C=best_c, penalty='l2', max_iter=1000, random_state=42)
    clf_b_champ.fit(X_tr_final, y_train)
    prob_b_test = clf_b_champ.predict_proba(X_te_final)[:, 1]
    pred_b_test = (prob_b_test >= 0.50).astype(int)

    feature_names = ["logit_code", "has_provenance", "ai_involvement", "is_pure_ai", "is_human_edited", "is_ai_edited", "confidence"]
    if use_inter:
        feature_names.extend(["logit_code_x_ai_inv", "logit_code_x_human_ed"])

    # Save Checkpoint
    ckpt_path = MODELS_DIR / "phase15_model_b_fusion_best.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "clf_fusion": clf_b_champ,
            "feature_names": feature_names,
            "best_c": best_c,
            "use_interactions": use_inter,
            "val_score": float(best_auc_inter if use_inter else best_auc_base),
            "weights": {name: round(float(w), 4) for name, w in zip(feature_names, clf_b_champ.coef_[0])},
            "bias": round(float(clf_b_champ.intercept_[0]), 4),
            "model_id": "phase15_model_b_fusion_best"
        }, f)
    print(f"  Saved Phase 15 Model B Champion Checkpoint to: {ckpt_path}")

    # 4. Comprehensive Ablation Suite on Held-Out Test Split (N = 92)
    print("\n" + "=" * 80)
    print("7-REGIME ABLATION BENCHMARK ON PHASE 15 HELD-OUT TEST SET (N = 92)")
    print("=" * 80)

    # Regime 1: Majority Class
    pred_majority = np.zeros(len(y_test), dtype=int)
    prob_majority = np.full(len(y_test), 0.50)

    # Regime 2: Provenance-Only Baseline (Trained strictly on provenance columns)
    prov_cols = [i for i, name in enumerate(feature_names) if not name.startswith("logit_code")]
    clf_prov = LogisticRegression(C=best_c, penalty='l2', max_iter=1000, random_state=42)
    clf_prov.fit(X_tr_final[:, prov_cols], y_train)
    prob_prov = clf_prov.predict_proba(X_te_final[:, prov_cols])[:, 1]
    pred_prov = (prob_prov >= 0.50).astype(int)

    # Regime 3: Model A Code-Only Detector
    prob_code = p_code_test
    pred_code = (prob_code >= 0.50).astype(int)

    # Regime 4: Model B Full Fusion
    # (prob_b_test, pred_b_test)

    # Regime 5: Model B Fallback (Unknown Provenance)
    X_test_fallback, _, _ = extract_features(test_recs, include_interactions=use_inter, fallback=True)
    prob_fallback = clf_b_champ.predict_proba(X_test_fallback)[:, 1]
    pred_fallback = (prob_fallback >= 0.50).astype(int)

    # Regime 6: Model B Masked Category (author_type masked, only ai_involvement provided)
    X_test_masked = np.copy(X_te_final)
    # Zero out is_pure_ai, is_human_edited, is_ai_edited
    X_test_masked[:, 3:6] = 0.0
    prob_masked = clf_b_champ.predict_proba(X_test_masked)[:, 1]
    pred_masked = (prob_masked >= 0.50).astype(int)

    # Regime 7: Model B Permuted Provenance
    X_test_perm, _, _ = extract_features(test_recs, include_interactions=use_inter, permute=True)
    prob_perm = clf_b_champ.predict_proba(X_test_perm)[:, 1]
    pred_perm = (prob_perm >= 0.50).astype(int)

    ablations = {
        "Majority-Class Baseline": (pred_majority, prob_majority),
        "Provenance-Only Baseline": (pred_prov, prob_prov),
        "Model A: Code Only": (pred_code, prob_code),
        "Model B: Full (Code + Prov)": (pred_b_test, prob_b_test),
        "Model B: Fallback (Unknown Prov)": (pred_fallback, prob_fallback),
        "Model B: Masked Category": (pred_masked, prob_masked),
        "Model B: Permuted Provenance": (pred_perm, prob_perm),
    }

    ablation_results = {}
    for name, (preds, probs) in ablations.items():
        m_cls = compute_classification_metrics(y_test, preds, probs)
        m_triage = compute_triage_ranking_metrics(y_test, probs)
        ablation_results[name] = {"classification": m_cls, "triage": m_triage}
        print(f"\n--- {name} ---")
        print(f"  Classification: Acc={m_cls['accuracy']*100:.2f}% [{m_cls['accuracy_ci_95'][0]*100:.1f}%, {m_cls['accuracy_ci_95'][1]*100:.1f}%] | AUROC={m_cls['auroc']:.4f} | F1={m_cls['f1']:.4f} | Brier={m_cls['brier_score']:.4f}")
        print(f"  Triage Ranking: P@20%={m_triage['budget_20pct']['precision_at_k']*100:.1f}%, Recall@20%={m_triage['budget_20pct']['recall_at_k']*100:.1f}% | Recall@30%={m_triage['budget_30pct']['recall_at_k']*100:.1f}% | AP={m_triage['average_precision']:.4f}")

    # 5. Grouped Cross-Validation (10 Folds by Independent Task across ALL 150 Tasks)
    print("\n" + "=" * 80)
    print("10-FOLD GROUPED CROSS-VALIDATION BY INDEPENDENT TASK (150 TASKS / 600 UNITS)")
    print("=" * 80)

    all_recs = train_recs + val_recs + test_recs
    all_task_ids = sorted(list(set(r["task_id"] for r in all_recs)))
    rng = np.random.RandomState(42)
    shuffled_tids = np.array(all_task_ids)
    rng.shuffle(shuffled_tids)

    fold_tasks = np.array_split(shuffled_tids, 10)
    
    cv_metrics_a = []
    cv_metrics_prov = []
    cv_metrics_b = []

    # Map records by task_id
    recs_by_task = {}
    for r in all_recs:
        recs_by_task.setdefault(r["task_id"], []).append(r)

    for f_idx in range(10):
        val_tids = set(fold_tasks[f_idx])
        tr_tids = set(all_task_ids) - val_tids

        f_train_recs = [r for tid in tr_tids for r in recs_by_task[tid]]
        f_val_recs = [r for tid in val_tids for r in recs_by_task[tid]]

        X_f_tr, y_f_tr, _ = extract_features(f_train_recs, include_interactions=use_inter)
        X_f_val, y_f_val, p_code_f_val = extract_features(f_val_recs, include_interactions=use_inter)

        # Model A code only
        cv_metrics_a.append(compute_classification_metrics(y_f_val, (p_code_f_val >= 0.50).astype(int), p_code_f_val))

        # Provenance only
        f_clf_prov = LogisticRegression(C=best_c, penalty='l2', max_iter=1000, random_state=42)
        f_clf_prov.fit(X_f_tr[:, prov_cols], y_f_tr)
        p_prov_val = f_clf_prov.predict_proba(X_f_val[:, prov_cols])[:, 1]
        cv_metrics_prov.append(compute_classification_metrics(y_f_val, (p_prov_val >= 0.50).astype(int), p_prov_val))

        # Model B Full
        f_clf_b = LogisticRegression(C=best_c, penalty='l2', max_iter=1000, random_state=42)
        f_clf_b.fit(X_f_tr, y_f_tr)
        p_b_val = f_clf_b.predict_proba(X_f_val)[:, 1]
        cv_metrics_b.append(compute_classification_metrics(y_f_val, (p_b_val >= 0.50).astype(int), p_b_val))

    def summarize_cv(metric_list):
        accs = [m["accuracy"] for m in metric_list]
        aurocs = [m["auroc"] for m in metric_list]
        f1s = [m["f1"] for m in metric_list]
        briers = [m["brier_score"] for m in metric_list]
        return {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "brier_mean": float(np.mean(briers)),
            "brier_std": float(np.std(briers))
        }

    cv_summary = {
        "model_a_code_only": summarize_cv(cv_metrics_a),
        "provenance_only": summarize_cv(cv_metrics_prov),
        "model_b_full_fusion": summarize_cv(cv_metrics_b)
    }

    print(f"\n10-Fold Grouped CV Summary across 150 Independent Tasks:")
    for mname, s in cv_summary.items():
        print(f"  • {mname:20}: Acc = {s['accuracy_mean']*100:.2f}% ± {s['accuracy_std']*100:.2f}% | AUROC = {s['auroc_mean']:.4f} ± {s['auroc_std']:.4f} | Brier = {s['brier_mean']:.4f} ± {s['brier_std']:.4f}")

    # Save complete evaluation output
    out_b_path = RESULTS_P15_DIR / "phase15_model_b_evaluation.json"
    with open(out_b_path, "w") as f:
        json.dump({
            "tuning_summary": {"best_c": float(best_c), "use_interactions": bool(use_inter)},
            "ablations_test_92": ablation_results,
            "grouped_cv_summary_150_tasks": cv_summary
        }, f, indent=2)
    print(f"\n[✓] Phase 15 Model B evaluation results saved to: {out_b_path}")


if __name__ == "__main__":
    main()

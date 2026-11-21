#!/usr/bin/env python3
"""
scripts/audit/reproduce_model_b.py
==================================
Independent reproducibility and validity audit of Model B.
Evaluates phase15_model_b_fusion_best.pkl on the Model B test split:
- Report n_units, n_tasks, n_positive, AUROC with task-cluster bootstrap CI (1000 resamples), AUPRC, precision@20% as k/n.
- Trains and evaluates 4 quick ablation models on the same split:
  (i) P_code only
  (ii) provenance features only (no P_code)
  (iii) author_role one-hot only
  (iv) existing fusion
  Reports AUROC with task-level bootstrap CI, plus DeLong test of (iv) vs (i).
- Computes label rate by author_role for train, val, and test.
- Computes provenance feature value ranges and flags constant/missing features.
- Tests fallback claim with missing provenance: max |P_risk - P_code|.
"""

import sys
import os
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_P15_DIR = PROJECT_ROOT / "data" / "phase15_model_b"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_P15_DIR = PROJECT_ROOT / "results" / "phase15"
RESULTS_AUDIT_DIR = PROJECT_ROOT / "results" / "audit"
RESULTS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

from src.models.inference import predict_model_b, set_active_checkpoints


def delong_roc_test(y_true: np.ndarray, prob1: np.ndarray, prob2: np.ndarray) -> Dict[str, Any]:
    """Computes DeLong test for paired AUROCs."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    d1 = prob1[pos_idx][:, None] - prob1[neg_idx][None, :]
    psi1 = (d1 > 0).astype(float) + 0.5 * (d1 == 0).astype(float)
    v10 = np.mean(psi1, axis=1)
    v01 = np.mean(psi1, axis=0)
    auc1 = float(np.mean(v10))

    d2 = prob2[pos_idx][:, None] - prob2[neg_idx][None, :]
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
        "auc1": auc1,
        "auc2": auc2,
        "delta": float(auc1 - auc2),
        "z_score": float(z),
        "p_value": float(p_val),
        "significant_p05": bool(p_val < 0.05)
    }


def task_bootstrap_auroc_ci(
    tasks: np.ndarray,
    y_true: np.ndarray,
    prob: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42
) -> List[float]:
    """Computes 95% bootstrap CI by resampling independent TASKS, not units."""
    unique_tasks = np.unique(tasks)
    n_tasks = len(unique_tasks)
    rng = np.random.RandomState(seed)
    boot_aucs = []

    for _ in range(n_resamples):
        sampled_tasks = rng.choice(unique_tasks, size=n_tasks, replace=True)
        b_indices = []
        for t in sampled_tasks:
            b_indices.extend(np.where(tasks == t)[0])
        b_y = y_true[b_indices]
        b_prob = prob[b_indices]
        if len(np.unique(b_y)) > 1:
            boot_aucs.append(roc_auc_score(b_y, b_prob))

    if not boot_aucs:
        return [0.5, 0.5]
    return [float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))]


def extract_feature_variants(recs: List[Dict[str, Any]], p_code_cache: Dict[str, float], use_interactions: bool = False):
    """Extracts features for all 4 ablation regimes."""
    X_pcode, X_prov, X_role, X_full, y, tasks = [], [], [], [], [], []
    eps = 1e-4

    for r in recs:
        c = r["code"]
        p_c = p_code_cache.get(c, 0.5)
        l_c = float(np.log((p_c + eps) / (1.0 - p_c + eps)))

        ptype = r.get("author_type", "human")
        ai_inv = float(r.get("ai_involvement", 0.0))
        conf = float(r.get("provenance_confidence", 1.0))

        is_human = 1.0 if ptype == "human" else 0.0
        is_pure_ai = 1.0 if ptype == "ai" else 0.0
        is_human_edited = 1.0 if ptype == "human_edit_of_ai" else 0.0
        is_ai_edited = 1.0 if ptype == "ai_edit_of_human" else 0.0

        # (i) P_code only
        X_pcode.append([l_c])

        # (ii) Provenance only (no P_code)
        X_prov.append([1.0, ai_inv, is_pure_ai, is_human_edited, is_ai_edited, conf])

        # (iii) Author role one-hot only
        X_role.append([is_human, is_pure_ai, is_human_edited, is_ai_edited])

        # (iv) Full fusion
        feat = [l_c, 1.0, ai_inv, is_pure_ai, is_human_edited, is_ai_edited, conf]
        if use_interactions:
            feat.extend([l_c * ai_inv, l_c * is_human_edited])
        X_full.append(feat)

        y.append(int(r["vuln_label"]))
        tasks.append(r["task_id"])

    return (
        np.array(X_pcode, dtype=float),
        np.array(X_prov, dtype=float),
        np.array(X_role, dtype=float),
        np.array(X_full, dtype=float),
        np.array(y, dtype=int),
        np.array(tasks)
    )


def main():
    t_start = time.time()
    print("=" * 80)
    print("AUDIT: MODEL B INDEPENDENT REPRODUCIBILITY & ABLATION EVALUATION")
    print("=" * 80)

    # 1. Load Model B Partitions
    print("\n[1] Loading Model B Partitions from data/phase15_model_b/...")
    with open(DATA_P15_DIR / "model_b_p15_train.json") as f:
        train_recs = json.load(f)["records"]
    with open(DATA_P15_DIR / "model_b_p15_val.json") as f:
        val_recs = json.load(f)["records"]
    with open(DATA_P15_DIR / "model_b_p15_test.json") as f:
        test_recs = json.load(f)["records"]

    with open(RESULTS_P15_DIR / "p_code_cache.json") as f:
        p_code_cache = json.load(f)

    # 2. Reload Model B Checkpoint
    ckpt_path = MODELS_DIR / "phase15_model_b_fusion_best.pkl"
    print(f"\n[2] Reloading Checkpoint: {ckpt_path.name}")
    with open(ckpt_path, "rb") as f:
        ckpt_b = pickle.load(f)

    clf_b_champ = ckpt_b["clf_fusion"]
    best_c = ckpt_b["best_c"]
    use_inter = ckpt_b.get("use_interactions", False)

    # 3. Extract Features across train, test
    (
        X_tr_pcode, X_tr_prov, X_tr_role, X_tr_full, y_tr, tasks_tr
    ) = extract_feature_variants(train_recs, p_code_cache, use_interactions=use_inter)

    (
        X_te_pcode, X_te_prov, X_te_role, X_te_full, y_te, tasks_te
    ) = extract_feature_variants(test_recs, p_code_cache, use_interactions=use_inter)

    # B1. Evaluate Existing Fusion Checkpoint
    print("\n[3] Re-evaluating Phase 15 Model B Champion on Test Split...")
    prob_b_test = clf_b_champ.predict_proba(X_te_full)[:, 1]
    pred_b_test = (prob_b_test >= 0.50).astype(int)

    n_units = len(y_te)
    n_tasks = len(np.unique(tasks_te))
    n_pos = int(np.sum(y_te == 1))

    auroc_b = float(roc_auc_score(y_te, prob_b_test))
    auprc_b = float(average_precision_score(y_te, prob_b_test))
    auroc_ci_b = task_bootstrap_auroc_ci(tasks_te, y_te, prob_b_test, n_resamples=1000, seed=42)

    # Precision@20%
    n_20 = max(1, int(n_units * 0.20))
    order_20 = np.argsort(-prob_b_test)
    k_20 = int(np.sum(y_te[order_20[:n_20]]))
    prec_20 = float(k_20 / n_20)

    b1_results = {
        "n_units": n_units,
        "n_tasks": n_tasks,
        "n_positive": n_pos,
        "auroc": auroc_b,
        "auroc_ci_95_task_bootstrap": auroc_ci_b,
        "auprc": auprc_b,
        "precision_at_20pct": {
            "k_caught": k_20,
            "n_inspected": n_20,
            "precision": prec_20
        }
    }
    print(f"  n_units: {n_units}, n_tasks: {n_tasks}, n_positive: {n_pos}")
    print(f"  AUROC: {auroc_b:.4f} (95% Task-Bootstrap CI: [{auroc_ci_b[0]:.4f}, {auroc_ci_b[1]:.4f}])")
    print(f"  AUPRC: {auprc_b:.4f}")
    print(f"  Precision@20%: {k_20}/{n_20} ({prec_20*100:.2f}%)")

    # B2. Train and Evaluate Ablation Models
    print("\n[4] Training and Evaluating Ablation Models...")

    # (i) P_code only
    clf_i = LogisticRegression(C=best_c, penalty='l2', max_iter=1000, random_state=42)
    clf_i.fit(X_tr_pcode, y_tr)
    prob_i = clf_i.predict_proba(X_te_pcode)[:, 1]
    auc_i = float(roc_auc_score(y_te, prob_i))
    ci_i = task_bootstrap_auroc_ci(tasks_te, y_te, prob_i, n_resamples=1000, seed=42)

    # (ii) Provenance only
    clf_ii = LogisticRegression(C=best_c, penalty='l2', max_iter=1000, random_state=42)
    clf_ii.fit(X_tr_prov, y_tr)
    prob_ii = clf_ii.predict_proba(X_te_prov)[:, 1]
    auc_ii = float(roc_auc_score(y_te, prob_ii))
    ci_ii = task_bootstrap_auroc_ci(tasks_te, y_te, prob_ii, n_resamples=1000, seed=42)

    # (iii) Author role only
    clf_iii = LogisticRegression(C=best_c, penalty='l2', max_iter=1000, random_state=42)
    clf_iii.fit(X_tr_role, y_tr)
    prob_iii = clf_iii.predict_proba(X_te_role)[:, 1]
    auc_iii = float(roc_auc_score(y_te, prob_iii))
    ci_iii = task_bootstrap_auroc_ci(tasks_te, y_te, prob_iii, n_resamples=1000, seed=42)

    # (iv) Existing fusion (already computed as prob_b_test)
    prob_iv = prob_b_test
    auc_iv = auroc_b
    ci_iv = auroc_ci_b

    # DeLong (iv) vs (i)
    delong_res = delong_roc_test(y_te, prob_iv, prob_i)

    b2_results = {
        "models": {
            "(i) P_code only": {"auroc": auc_i, "auroc_ci_95": ci_i},
            "(ii) Provenance features only": {"auroc": auc_ii, "auroc_ci_95": ci_ii},
            "(iii) Author role one-hot only": {"auroc": auc_iii, "auroc_ci_95": ci_iii},
            "(iv) Existing fusion": {"auroc": auc_iv, "auroc_ci_95": ci_iv}
        },
        "delong_iv_vs_i": delong_res
    }
    print(f"  (i) P_code only: AUROC = {auc_i:.4f} (CI: [{ci_i[0]:.4f}, {ci_i[1]:.4f}])")
    print(f"  (ii) Provenance only: AUROC = {auc_ii:.4f} (CI: [{ci_ii[0]:.4f}, {ci_ii[1]:.4f}])")
    print(f"  (iii) Author role only: AUROC = {auc_iii:.4f} (CI: [{ci_iii[0]:.4f}, {ci_iii[1]:.4f}])")
    print(f"  (iv) Existing fusion: AUROC = {auc_iv:.4f} (CI: [{ci_iv[0]:.4f}, {ci_iv[1]:.4f}])")
    print(f"  DeLong (iv) vs (i): Delta = {delong_res['delta']:+.4f}, z = {delong_res['z_score']:.4f}, p = {delong_res['p_value']:.4e}")

    # B4. Label Rate by Author Role across Splits
    print("\n[5] Computing Label Rate by Author Role for Train, Val, Test...")
    splits_dict = {"train": train_recs, "val": val_recs, "test": test_recs}
    roles = ["human", "ai", "human_edit_of_ai", "ai_edit_of_human"]
    b4_results = {}

    for sname, recs in splits_dict.items():
        b4_results[sname] = {}
        for r in roles:
            sub = [x for x in recs if x.get("author_type") == r]
            n_sub = len(sub)
            n_v = sum(1 for x in sub if x["vuln_label"] == 1)
            rate = float(n_v / n_sub) if n_sub > 0 else 0.0
            b4_results[sname][r] = {
                "n": n_sub,
                "n_vulnerable": n_v,
                "vuln_rate": rate,
                "pct": round(rate * 100.0, 2)
            }

    # B5. Feature Value Range per Author Role & Flagging Constants
    print("\n[6] Computing Provenance Feature Value Ranges and Flagging Constants...")
    all_recs = train_recs + val_recs + test_recs
    b5_results = {}
    prov_features_to_check = [
        "ai_involvement",
        "human_edit_count",
        "provenance_confidence",
        "is_pure_ai",
        "is_human_edited",
        "is_ai_edited"
    ]

    for feat in prov_features_to_check:
        b5_results[feat] = {}
        for r in roles:
            vals = []
            for item in all_recs:
                if item.get("author_type") == r:
                    if feat == "is_pure_ai":
                        vals.append(1.0 if item.get("author_type") == "ai" else 0.0)
                    elif feat == "is_human_edited":
                        vals.append(1.0 if item.get("author_type") == "human_edit_of_ai" else 0.0)
                    elif feat == "is_ai_edited":
                        vals.append(1.0 if item.get("author_type") == "ai_edit_of_human" else 0.0)
                    else:
                        vals.append(float(item.get(feat, 0.0)))
            vals = np.array(vals)
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            unique_v = sorted(list(set(vals.tolist())))
            is_constant = (len(unique_v) <= 1)
            b5_results[feat][r] = {
                "min": min_v,
                "max": max_v,
                "unique_values": unique_v,
                "is_constant": is_constant,
                "flag": "CONSTANT" if is_constant else "VARIABLE"
            }

    # B6. Fallback Claim Verification: max |P_risk - P_code|
    print("\n[7] Testing Model B Fallback Claim on Test Units...")
    set_active_checkpoints(model_a_name="expanded_tri_union_best.pkl", model_b_name="phase15_model_b_fusion_best.pkl")
    max_fallback_diff = 0.0
    for r in test_recs:
        res = predict_model_b(r["code"], provenance=None)
        p_code = res["base_code_probability"]
        p_risk = res["probability"]
        diff = abs(p_risk - p_code)
        if diff > max_fallback_diff:
            max_fallback_diff = diff

    b6_results = {
        "n_units_tested": len(test_recs),
        "max_abs_diff": float(max_fallback_diff),
        "claim_verified": bool(max_fallback_diff < 1e-6)
    }
    print(f"  Tested {len(test_recs)} test units with provenance=None.")
    print(f"  Max |P_risk - P_code| = {max_fallback_diff:.8f}")

    # Save output JSON
    total_runtime = time.time() - t_start
    output_payload = {
        "metadata": {
            "task": "Model B Reproducibility and Validity Audit",
            "runtime_seconds": round(total_runtime, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "b1_reproduction": b1_results,
        "b2_ablation": b2_results,
        "b4_label_rate_by_role": b4_results,
        "b5_feature_ranges": b5_results,
        "b6_fallback_test": b6_results
    }

    out_file = RESULTS_AUDIT_DIR / "model_b_reproduction.json"
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[✓] Model B Audit successfully completed in {total_runtime:.2f}s!")
    print(f"[✓] Results written to: {out_file}")


if __name__ == "__main__":
    main()

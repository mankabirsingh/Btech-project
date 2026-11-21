#!/usr/bin/env python3
"""
scripts/p7/train_eval_model_b.py
================================
Phase 16 Stage 2 / Task 10: Part C Model B Evaluation and Offset Fusion Experiments.

Evaluates 5 models under 5-fold GroupKFold by task across 5 seeds (42, 43, 44, 45, 46):
- M1: P_code from models/checkpoints/p5/model_a_final/predictor.py (predict) only.
- M2: Provenance features only.
- M3: Offset fusion model: logit(P_risk) = logit(P_code) + w * provenance (fixed offset).
      Unit test: if provenance missing/zero, P_risk == P_code +/- 10^-6.
- M4: M3 with provenance shuffled within each task (negative control).
- M5: Static-tool finding count only.

Metrics: AUROC, AUPRC, precision@20%.
Paired task-grouped bootstrap tests (2000 resamples by task):
- M3 vs M1
- M3 vs M4
- M1 vs M5
Raw and Holm-adjusted p-values.

Repeats across:
1. Primary: oracle_label (600 units)
2. Repeat (a): orig_label (generator labels, circular, for comparison only) (600 units)
3. Repeat (b): oracle_label with human_edit_of_ai removed (450 units; human edits simulated)
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import scipy.special as sp
import scipy.optimize as opt
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.stats import format_p_value, holm_adjust

DATA_P7_DIR = PROJECT_ROOT / "data" / "p7"
RESULTS_P7_DIR = PROJECT_ROOT / "results" / "p7"
MODELS_P7_DIR = PROJECT_ROOT / "models" / "checkpoints" / "p7"

RESULTS_P7_DIR.mkdir(parents=True, exist_ok=True)
MODELS_P7_DIR.mkdir(parents=True, exist_ok=True)


def precision_at_20(y_true: np.ndarray, y_prob: np.ndarray, *args, **kwargs) -> float:
    """Computes precision at top 20% scores."""
    n = len(y_true)
    k = max(1, int(round(0.20 * n)))
    # For ranking, use stable sort descending
    top_indices = np.argsort(y_prob)[::-1][:k]
    return float(np.mean(y_true[top_indices]))


def fit_offset_logistic(X: np.ndarray, y: np.ndarray, offset: np.ndarray, C: float = 1.0) -> np.ndarray:
    """
    Fits logistic regression with a fixed offset and no intercept:
    logit(p) = offset + X @ w
    Loss = - sum(y * logits - log(1 + exp(logits))) + 0.5/C * ||w||^2
    Guarantees: when X == 0, logit(p) == offset, so p == expit(offset) == P_code.
    """
    n_features = X.shape[1]

    def loss_and_grad(w):
        logits = offset + X @ w
        p = sp.expit(logits)
        ll = - np.sum(y * logits - np.logaddexp(0, logits))
        reg = 0.5 / C * np.sum(w ** 2)
        grad = X.T @ (p - y) + (1.0 / C) * w
        return ll + reg, grad

    w0 = np.zeros(n_features)
    res = opt.minimize(loss_and_grad, w0, jac=True, method="L-BFGS-B")
    return res.x


def verify_offset_unit_test(w: np.ndarray, p_code_samples: np.ndarray, n_features: int):
    """Verifies that if provenance is missing/zero, P_risk == P_code +/- 10^-6."""
    offset = sp.logit(p_code_samples)
    zero_prov = np.zeros((len(p_code_samples), n_features))
    logits = offset + zero_prov @ w
    p_risk = sp.expit(logits)
    max_err = np.max(np.abs(p_risk - p_code_samples))
    assert max_err <= 1e-6, f"Unit test failed! max_err = {max_err}"
    return float(max_err)


def compute_task_bootstrap_ci(
    tasks: np.ndarray,
    y_true: np.ndarray,
    prob: np.ndarray,
    metric_fn,
    n_resamples: int = 2000,
    seed: int = 42
) -> List[float]:
    """Computes 95% bootstrap CI by resampling independent TASKS with replacement."""
    unique_tasks = np.unique(tasks)
    n_tasks = len(unique_tasks)
    task_to_idx = {t: np.where(tasks == t)[0] for t in unique_tasks}
    rng = np.random.RandomState(seed)
    scores = []

    for _ in range(n_resamples):
        chosen_tasks = rng.choice(unique_tasks, size=n_tasks, replace=True)
        b_idx = np.concatenate([task_to_idx[t] for t in chosen_tasks])
        y_b = y_true[b_idx]
        if len(np.unique(y_b)) < 2:
            continue
        p_b = prob[b_idx]
        try:
            val = metric_fn(y_b, p_b)
            scores.append(val)
        except Exception:
            continue

    if not scores:
        return [0.0, 0.0]
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def compute_paired_task_bootstrap(
    tasks: np.ndarray,
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    metric_fn,
    n_resamples: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes paired bootstrap test for Delta = metric(Model B) - metric(Model A)
    resampling clustered by task.
    """
    obs_a = float(metric_fn(y_true, prob_a))
    obs_b = float(metric_fn(y_true, prob_b))
    diff_obs = float(obs_b - obs_a)

    unique_tasks = np.unique(tasks)
    n_tasks = len(unique_tasks)
    task_to_idx = {t: np.where(tasks == t)[0] for t in unique_tasks}
    rng = np.random.RandomState(seed)
    diffs = []

    for _ in range(n_resamples):
        chosen_tasks = rng.choice(unique_tasks, size=n_tasks, replace=True)
        b_idx = np.concatenate([task_to_idx[t] for t in chosen_tasks])
        y_b = y_true[b_idx]
        if len(np.unique(y_b)) < 2:
            continue
        pa_b = prob_a[b_idx]
        pb_b = prob_b[b_idx]
        try:
            val_a = metric_fn(y_b, pa_b)
            val_b = metric_fn(y_b, pb_b)
            diffs.append(val_b - val_a)
        except Exception:
            continue

    if not diffs:
        return {
            "diff_observed": diff_obs,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
            "p_value_formatted": "1.000"
        }

    diffs_arr = np.array(diffs)
    ci_lower = float(np.percentile(diffs_arr, 2.5))
    ci_upper = float(np.percentile(diffs_arr, 97.5))

    frac_le_zero = np.mean(diffs_arr <= 0)
    frac_ge_zero = np.mean(diffs_arr >= 0)
    p_val = float(min(1.0, 2.0 * min(frac_le_zero, frac_ge_zero)))

    return {
        "score_a": obs_a,
        "score_b": obs_b,
        "diff_observed": diff_obs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "p_value_formatted": format_p_value(p_val, 1.0 / n_resamples)
    }


def run_experiment_setting(
    name: str,
    records: List[Dict[str, Any]],
    p_code: np.ndarray,
    label_key: str,
    seeds: List[int] = [42, 43, 44, 45, 46]
) -> Dict[str, Any]:
    print(f"\n--- Running Experiment: {name} (N={len(records)}, label={label_key}) ---")

    tasks = np.array([r["task_id"] for r in records])
    unique_tasks = np.unique(tasks)
    y_true = np.array([r[label_key] for r in records], dtype=int)
    counts = np.array([r["total_findings_count"] for r in records], dtype=float)

    # Feature extraction
    X_prov = []
    for r in records:
        ptype = r.get("author_type", "human")
        ai_inv = float(r.get("ai_involvement", 0.0))
        conf = float(r.get("provenance_confidence", 1.0))
        edits = float(r.get("human_edit_count", 0))
        is_pure_ai = 1.0 if ptype == "ai" else 0.0
        is_human_edited = 1.0 if ptype == "human_edit_of_ai" else 0.0
        is_ai_edited = 1.0 if ptype == "ai_edit_of_human" else 0.0
        X_prov.append([ai_inv, is_pure_ai, is_human_edited, is_ai_edited, conf, edits])
    X_prov = np.array(X_prov, dtype=float)

    model_names = ["M1_P_code", "M2_Provenance_only", "M3_Offset_fusion", "M4_Provenance_shuffled", "M5_Static_count"]

    # Storage for out-of-fold predictions across seeds
    oof_preds = {m: np.zeros((len(seeds), len(records))) for m in model_names}
    seed_metrics = {m: {"auroc": [], "auprc": [], "prec_at_20": []} for m in model_names}

    for s_idx, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        shuffled_tasks = rng.permutation(unique_tasks)
        folds = np.array_split(shuffled_tasks, 5)

        p_m1 = p_code.copy()
        p_m2 = np.zeros(len(records))
        p_m3 = np.zeros(len(records))
        p_m4 = np.zeros(len(records))
        p_m5 = np.zeros(len(records))

        # Provenance shuffled control matrix for this seed
        X_prov_shuff = X_prov.copy()
        for t in unique_tasks:
            t_idx = np.where(tasks == t)[0]
            X_prov_shuff[t_idx] = rng.permutation(X_prov_shuff[t_idx])

        for test_tasks in folds:
            test_mask = np.isin(tasks, test_tasks)
            train_mask = ~test_mask

            # M2: Provenance only
            clf_m2 = LogisticRegression(C=1.0, max_iter=1000)
            clf_m2.fit(X_prov[train_mask], y_true[train_mask])
            p_m2[test_mask] = clf_m2.predict_proba(X_prov[test_mask])[:, 1]

            # M3: Offset fusion model
            offset_tr = sp.logit(p_code[train_mask])
            w_m3 = fit_offset_logistic(X_prov[train_mask], y_true[train_mask], offset_tr)
            offset_te = sp.logit(p_code[test_mask])
            p_m3[test_mask] = sp.expit(offset_te + X_prov[test_mask] @ w_m3)

            # M4: M3 with shuffled provenance
            offset_tr = sp.logit(p_code[train_mask])
            w_m4 = fit_offset_logistic(X_prov_shuff[train_mask], y_true[train_mask], offset_tr)
            p_m4[test_mask] = sp.expit(offset_te + X_prov_shuff[test_mask] @ w_m4)

            # M5: Static tool finding count only
            clf_m5 = LogisticRegression(C=1.0, max_iter=1000)
            clf_m5.fit(counts[train_mask].reshape(-1, 1), y_true[train_mask])
            p_m5[test_mask] = clf_m5.predict_proba(counts[test_mask].reshape(-1, 1))[:, 1]

        preds_seed = {
            "M1_P_code": p_m1,
            "M2_Provenance_only": p_m2,
            "M3_Offset_fusion": p_m3,
            "M4_Provenance_shuffled": p_m4,
            "M5_Static_count": p_m5
        }

        for m in model_names:
            p = preds_seed[m]
            oof_preds[m][s_idx] = p
            seed_metrics[m]["auroc"].append(roc_auc_score(y_true, p))
            seed_metrics[m]["auprc"].append(average_precision_score(y_true, p))
            seed_metrics[m]["prec_at_20"].append(precision_at_20(y_true, p))

    # Mean pooled predictions across seeds
    mean_preds = {m: np.mean(oof_preds[m], axis=0) for m in model_names}

    # Evaluate mean metrics and 95% bootstrap CIs
    models_report = {}
    for m in model_names:
        p = mean_preds[m]
        auroc_obs = float(roc_auc_score(y_true, p))
        auprc_obs = float(average_precision_score(y_true, p))
        prec20_obs = float(precision_at_20(y_true, p))

        auroc_ci = compute_task_bootstrap_ci(tasks, y_true, p, roc_auc_score, seed=42)
        auprc_ci = compute_task_bootstrap_ci(tasks, y_true, p, average_precision_score, seed=42)
        prec20_ci = compute_task_bootstrap_ci(tasks, y_true, p, precision_at_20, seed=42)

        models_report[m] = {
            "auroc_mean": float(np.mean(seed_metrics[m]["auroc"])),
            "auroc_std": float(np.std(seed_metrics[m]["auroc"])),
            "auroc_obs": auroc_obs,
            "auroc_ci": auroc_ci,
            "auprc_mean": float(np.mean(seed_metrics[m]["auprc"])),
            "auprc_std": float(np.std(seed_metrics[m]["auprc"])),
            "auprc_obs": auprc_obs,
            "auprc_ci": auprc_ci,
            "prec_at_20_mean": float(np.mean(seed_metrics[m]["prec_at_20"])),
            "prec_at_20_std": float(np.std(seed_metrics[m]["prec_at_20"])),
            "prec_at_20_obs": prec20_obs,
            "prec_at_20_ci": prec20_ci
        }

    # Paired comparisons: M3 vs M1, M3 vs M4, M1 vs M5
    pairs = [
        ("M3_vs_M1", "M1_P_code", "M3_Offset_fusion"),
        ("M3_vs_M4", "M4_Provenance_shuffled", "M3_Offset_fusion"),
        ("M1_vs_M5", "M5_Static_count", "M1_P_code")
    ]

    paired_results = {}
    raw_p_values = []
    pair_keys = []

    for pair_name, m_a, m_b in pairs:
        pa = mean_preds[m_a]
        pb = mean_preds[m_b]

        pair_data = {
            "model_a": m_a,
            "model_b": m_b,
            "auroc": compute_paired_task_bootstrap(tasks, y_true, pa, pb, roc_auc_score, seed=42),
            "auprc": compute_paired_task_bootstrap(tasks, y_true, pa, pb, average_precision_score, seed=42),
            "prec_at_20": compute_paired_task_bootstrap(tasks, y_true, pa, pb, precision_at_20, seed=42)
        }
        paired_results[pair_name] = pair_data
        raw_p_values.append(pair_data["auroc"]["p_value"])
        pair_keys.append(pair_name)

    # Holm-Bonferroni adjustment across the 3 primary AUROC comparisons
    holm_p_vals = holm_adjust(raw_p_values)
    for p_key, h_p in zip(pair_keys, holm_p_vals):
        paired_results[p_key]["auroc"]["holm_p_value"] = float(h_p)
        paired_results[p_key]["auroc"]["holm_p_formatted"] = format_p_value(h_p, 1.0 / 2000)

    # Print summary table
    print("\nModel Performance Summary:")
    print(f"{'Model':25s} | {'AUROC [95% CI]':28s} | {'AUPRC [95% CI]':28s} | {'Prec@20% [95% CI]':28s}")
    print("-" * 115)
    for m in model_names:
        r = models_report[m]
        print(f"{m:25s} | {r['auroc_obs']:.4f} [{r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f}] | "
              f"{r['auprc_obs']:.4f} [{r['auprc_ci'][0]:.4f}, {r['auprc_ci'][1]:.4f}] | "
              f"{r['prec_at_20_obs']:.4f} [{r['prec_at_20_ci'][0]:.4f}, {r['prec_at_20_ci'][1]:.4f}]")

    print("\nPaired Comparisons (AUROC):")
    for pair_name in pair_keys:
        res = paired_results[pair_name]["auroc"]
        print(f"  {pair_name:12s}: diff = {res['diff_observed']:+.4f} [{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}], "
              f"raw p = {res['p_value_formatted']}, Holm p = {res['holm_p_formatted']}")

    return {
        "setting_name": name,
        "n_units": len(records),
        "n_tasks": len(unique_tasks),
        "prevalence": float(np.mean(y_true)),
        "models": models_report,
        "paired_tests": paired_results
    }


def main():
    print("=" * 80)
    print("PART C: MODEL B REBUILD AND OFFSET FUSION EXPERIMENTS")
    print("=" * 80)
    t0 = time.time()

    # Load oracle labeled dataset
    with open(DATA_P7_DIR / "model_b_oracle_labels.json") as f:
        all_records = json.load(f)["records"]

    p_code = np.load(DATA_P7_DIR / "p_code_model_a_final.npy")
    assert len(all_records) == len(p_code), "Record count and P_code mismatch!"
    print(f"[1] Loaded {len(all_records)} units and aligned Model A Final P_code.")

    # 1. Primary setting: oracle labels (all 600 units)
    res_primary = run_experiment_setting(
        name="Primary: Oracle Labels",
        records=all_records,
        p_code=p_code,
        label_key="oracle_label"
    )

    # 2. Repeat (a): circular generator labels (all 600 units)
    res_orig = run_experiment_setting(
        name="Repeat A: Generator Labels (Circular, for comparison only)",
        records=all_records,
        p_code=p_code,
        label_key="orig_label"
    )

    # 3. Repeat (b): oracle labels with human_edit_of_ai removed (450 units)
    keep_indices = [i for i, r in enumerate(all_records) if r["author_type"] != "human_edit_of_ai"]
    sub_records = [all_records[i] for i in keep_indices]
    sub_p_code = p_code[keep_indices]
    print(f"\n[2] Filtered dataset to {len(sub_records)} units (human_edit_of_ai removed; simulated human edits).")

    res_no_human_edit = run_experiment_setting(
        name="Repeat B: Oracle Labels without human_edit_of_ai (Simulated Human Edits Removed)",
        records=sub_records,
        p_code=sub_p_code,
        label_key="oracle_label"
    )

    # Save full results JSON
    out_results_path = RESULTS_P7_DIR / "part_c_experiments.json"
    full_experiments = {
        "primary_oracle": res_primary,
        "repeat_a_circular_orig": res_orig,
        "repeat_b_no_simulated_human_edits": res_no_human_edit,
        "total_runtime_s": float(time.time() - t0)
    }
    with open(out_results_path, "w") as f:
        json.dump(full_experiments, f, indent=2)
    print(f"\n[3] Full experiment results saved to {out_results_path}")

    # Fit final offset model on full dataset and verify unit test
    print("\n[4] Fitting and serializing final Model B Offset Fusion Checkpoint...")
    X_prov_full = []
    for r in all_records:
        ptype = r.get("author_type", "human")
        ai_inv = float(r.get("ai_involvement", 0.0))
        conf = float(r.get("provenance_confidence", 1.0))
        edits = float(r.get("human_edit_count", 0))
        is_pure_ai = 1.0 if ptype == "ai" else 0.0
        is_human_edited = 1.0 if ptype == "human_edit_of_ai" else 0.0
        is_ai_edited = 1.0 if ptype == "ai_edit_of_human" else 0.0
        X_prov_full.append([ai_inv, is_pure_ai, is_human_edited, is_ai_edited, conf, edits])
    X_prov_full = np.array(X_prov_full, dtype=float)
    y_oracle_full = np.array([r["oracle_label"] for r in all_records], dtype=int)
    offset_full = sp.logit(p_code)

    w_final = fit_offset_logistic(X_prov_full, y_oracle_full, offset_full, C=1.0)
    max_err = verify_offset_unit_test(w_final, p_code, n_features=X_prov_full.shape[1])
    print(f"    Final w: {w_final}")
    print(f"    Unit test max |P_risk - P_code| at zero provenance: {max_err:.10e} (<= 1e-6: True)")

    ckpt = {
        "model_name": "model_b_offset_fusion",
        "description": "Offset logistic fusion model for provenance-aware vulnerability risk scoring",
        "formula": "logit(P_risk) = logit(P_code) + w * provenance",
        "feature_names": ["ai_involvement", "is_pure_ai", "is_human_edited", "is_ai_edited", "provenance_confidence", "human_edit_count"],
        "weights": w_final.tolist(),
        "regularization_C": 1.0,
        "n_samples": len(all_records),
        "zero_provenance_max_diff": max_err,
        "unit_test_passed": bool(max_err <= 1e-6),
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    out_ckpt_path = MODELS_P7_DIR / "model_b_offset_fusion.pkl"
    with open(out_ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"    Checkpoint saved to {out_ckpt_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/p12/compute_fixes.py

Recomputes and verifies all 7 items required for STEP 1 of the final packaging task:
1. Every length-matched AUROC CI on test2-clean and old test (2,000 bootstrap resamples clustered by commit).
2. V3 and V6 three-way counts on test2-clean (must sum to N=1991), with precision and NPV.
3. The actual test2 and test2-clean project lists, N and prevalence.
4. Per-project test2-clean metrics from saved predictions (n, positives, AUROC with CI per model).
5. Old test N and prevalence (correcting 4,016 / 31.75% to 3,169 / 31.65%).
6. Juliet: N per split and every model's AUROC/accuracy per split from results/p10; flag >0.98 as "possible leak".
7. Model B: validated harness count, units per role, clean tasks (36), CV method (5-fold GroupKFold by task across 5 seeds), C1 counts & McNemar results, C2 AUROCs with CIs.

Writes results to results/p12/fixes.json.
"""

import os
import sys
import json
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.eval.stats_v2 import calc_length_matched_auroc, get_length_matched_pair_indices
from scripts.p2.code_normalizer import normalize_code_string

RESULTS_P12_DIR = BASE_DIR / "results" / "p12"
RESULTS_P12_DIR.mkdir(parents=True, exist_ok=True)


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cluster_bootstrap_lm_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tokens: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """
    Computes cluster bootstrap confidence interval for length-matched AUROC.
    Simultaneously resamples y_true, y_prob, and tokens clustered by group.
    """
    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    rng = np.random.RandomState(seed)

    scores = []
    for _ in range(n_resamples):
        chosen_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        b_idx = np.concatenate([group_to_idx[g] for g in chosen_groups])
        yt_b = y_true[b_idx]
        if len(np.unique(yt_b)) < 2:
            continue
        yp_b = y_prob[b_idx]
        tok_b = tokens[b_idx]
        score = calc_length_matched_auroc(yt_b, yp_b, tok_b, max_log_diff=0.1)
        scores.append(score)

    if not scores:
        return 0.5, 0.5

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(scores, lower_pct)), float(np.percentile(scores, upper_pct))


def cluster_bootstrap_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[float, float]:
    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    rng = np.random.RandomState(seed)

    scores = []
    for _ in range(n_resamples):
        chosen_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        b_idx = np.concatenate([group_to_idx[g] for g in chosen_groups])
        yt_b = y_true[b_idx]
        if len(np.unique(yt_b)) < 2:
            continue
        yp_b = y_prob[b_idx]
        scores.append(roc_auc_score(yt_b, yp_b))

    if not scores:
        return 0.5, 0.5

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(scores, lower_pct)), float(np.percentile(scores, upper_pct))


def main():
    print("=" * 80)
    print("P12: COMPUTE NUMBER FIXES AND AUDIT GROUND TRUTH")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # ITEM 1: Length-matched AUROC CIs on test2-clean and old_test
    # -------------------------------------------------------------------------
    print("\n--- ITEM 1: Recomputing Length-Matched AUROC CIs ---")
    # Load caches & models
    with open(BASE_DIR / "data" / "cache" / "p10" / "test2_features_cache.pkl", "rb") as f:
        t2_cache = pickle.load(f)
    t2_len = t2_cache["t2_len"]
    t2_stat = t2_cache["t2_stat"]
    t2_qwen = t2_cache["t2_qwen"]
    t2_cb = t2_cache["t2_cb"]

    with open(BASE_DIR / "data" / "cache" / "p10" / "old_test_features_cache.pkl", "rb") as f:
        old_cache = pickle.load(f)
    t_old_len = old_cache["t_old_len"]
    t_old_stat = old_cache["t_old_stat"]
    t_old_qwen = old_cache["t_old_qwen"]
    t_old_cb = old_cache["t_old_cb"]

    with open(BASE_DIR / "data" / "cache" / "p10" / "model_a_v2_candidates_cache.pkl", "rb") as f:
        cands_cache = pickle.load(f)
    cands = cands_cache["candidates"]

    with open(BASE_DIR / "models" / "checkpoints" / "p10" / "model_a_v2_best.pkl", "rb") as f:
        best_pkg = pickle.load(f)
    scaler_stat = best_pkg["scaler_static"]
    scaler_len = best_pkg["scaler_len"]
    m_v3 = best_pkg["model"]

    with open(BASE_DIR / "models" / "checkpoints" / "p10" / "model_a_v2_stacker.pkl", "rb") as f:
        m_v6 = pickle.load(f)

    # Predictions test2
    t2_len_s = scaler_len.transform(t2_len)
    t2_stat_s = scaler_stat.transform(t2_stat)
    tok_t2 = np.exp(t2_len[:, 0]) - 1.0

    p_l1_t2 = cands["L1"].predict_proba(t2_len_s[:, :1])[:, 1]
    p_champ_t2 = cands["champion_faithful"].predict_proba(t2_cb)[:, 1]
    p_stat_t2 = cands["static_only"].predict_proba(t2_stat_s)[:, 1]
    X_v3_t2 = np.hstack([t2_qwen, t2_cb, t2_stat_s])
    p_v3_t2 = m_v3.predict_proba(X_v3_t2)[:, 1]
    logit_v3_t2 = np.log(np.clip(p_v3_t2, 1e-6, 1 - 1e-6) / (1 - np.clip(p_v3_t2, 1e-6, 1 - 1e-6)))
    p_v6_t2 = m_v6.predict_proba(np.column_stack([logit_v3_t2, t2_len[:, :1]]))[:, 1]

    # Predictions old_test
    t_old_len_s = scaler_len.transform(t_old_len)
    t_old_stat_s = scaler_stat.transform(t_old_stat)
    tok_old = np.exp(t_old_len[:, 0]) - 1.0

    p_l1_old = cands["L1"].predict_proba(t_old_len_s[:, :1])[:, 1]
    p_champ_old = cands["champion_faithful"].predict_proba(t_old_cb)[:, 1]
    p_stat_old = cands["static_only"].predict_proba(t_old_stat_s)[:, 1]
    X_v3_old = np.hstack([t_old_qwen, t_old_cb, t_old_stat_s])
    p_v3_old = m_v3.predict_proba(X_v3_old)[:, 1]
    logit_v3_old = np.log(np.clip(p_v3_old, 1e-6, 1 - 1e-6) / (1 - np.clip(p_v3_old, 1e-6, 1 - 1e-6)))
    p_v6_old = m_v6.predict_proba(np.column_stack([logit_v3_old, t_old_len[:, :1]]))[:, 1]

    # Load splits for metadata
    df_test2 = pd.read_parquet(BASE_DIR / "data" / "p10" / "test2.parquet")
    df_train_minus_dev = pd.read_parquet(BASE_DIR / "data" / "splits" / "p2" / "train_minus_dev.parquet")
    df_dev = pd.read_parquet(BASE_DIR / "data" / "splits" / "p2" / "dev_heldout.parquet")
    df_old_test = pd.read_parquet(BASE_DIR / "data" / "splits" / "vulnerability_test_heldout_projects_clean.parquet")

    t2_norm_hashes = df_test2["code"].apply(normalize_code_string).apply(sha256_str)
    train_norm_hashes = set(df_train_minus_dev["code"].apply(normalize_code_string).apply(sha256_str))
    dev_norm_hashes = set(df_dev["code"].apply(normalize_code_string).apply(sha256_str))
    old_test_norm_hashes = set(df_old_test["code"].apply(normalize_code_string).apply(sha256_str))

    all_norm_excluded_indices = (
        set(df_test2.index[t2_norm_hashes.isin(train_norm_hashes)]) |
        set(df_test2.index[t2_norm_hashes.isin(dev_norm_hashes)]) |
        set(df_test2.index[t2_norm_hashes.isin(old_test_norm_hashes)])
    )
    mask_clean_norm = ~df_test2.index.isin(all_norm_excluded_indices)

    y_t2_clean = df_test2.loc[mask_clean_norm, "vuln_label"].values.astype(int)
    commits_t2_clean = df_test2.loc[mask_clean_norm, "commit_id"].values
    tok_t2_clean = tok_t2[mask_clean_norm]

    models_t2_clean = {
        "L1": p_l1_t2[mask_clean_norm],
        "Champion": p_champ_t2[mask_clean_norm],
        "Static-only": p_stat_t2[mask_clean_norm],
        "V3": p_v3_t2[mask_clean_norm],
        "V6": p_v6_t2[mask_clean_norm],
    }

    y_old = df_old_test["vuln_label"].values.astype(int)
    commits_old = df_old_test["commit_id"].values
    models_old = {
        "L1": p_l1_old,
        "Champion": p_champ_old,
        "Static-only": p_stat_old,
        "V3": p_v3_old,
        "V6": p_v6_old,
    }

    # Load old broken values from results/p11/
    with open(BASE_DIR / "results" / "p11" / "test2_clean_evaluation.json") as f:
        t2_clean_json = json.load(f)["test2_clean_norm"]
    with open(BASE_DIR / "results" / "p11" / "old_test_evaluation.json") as f:
        old_test_json = json.load(f)

    item1_fixes = {
        "test2_clean": {},
        "old_test": {},
    }

    print("Computing 2,000 cluster-bootstrap iterations for test2-clean...")
    for m_name, p_arr in models_t2_clean.items():
        lm_val = calc_length_matched_auroc(y_t2_clean, p_arr, tok_t2_clean)
        old_ci = t2_clean_json[m_name]["length_matched_auroc_ci"]
        new_ci = list(cluster_bootstrap_lm_auroc(y_t2_clean, p_arr, tok_t2_clean, commits_t2_clean, n_resamples=2000, seed=42))
        item1_fixes["test2_clean"][m_name] = {
            "lm_auroc": lm_val,
            "old_ci": old_ci,
            "new_ci": new_ci,
            "status": "not distinguishable from chance" if (new_ci[0] <= 0.5 <= new_ci[1]) else ("below chance" if new_ci[1] < 0.5 else "above chance"),
            "source": "recomputed from data/cache/p10/test2_features_cache.pkl with simultaneous token resampling",
        }
        print(f"  test2-clean | {m_name:<12} | LM-AUROC: {lm_val:.4f} | Old CI: [{old_ci[0]:.4f}, {old_ci[1]:.4f}] -> New CI: [{new_ci[0]:.4f}, {new_ci[1]:.4f}]")

    print("Computing 2,000 cluster-bootstrap iterations for old_test...")
    for m_name, p_arr in models_old.items():
        lm_val = calc_length_matched_auroc(y_old, p_arr, tok_old)
        old_ci = old_test_json[m_name]["length_matched_auroc_ci"]
        new_ci = list(cluster_bootstrap_lm_auroc(y_old, p_arr, tok_old, commits_old, n_resamples=2000, seed=42))
        item1_fixes["old_test"][m_name] = {
            "lm_auroc": lm_val,
            "old_ci": old_ci,
            "new_ci": new_ci,
            "status": "not distinguishable from chance" if (new_ci[0] <= 0.5 <= new_ci[1]) else ("below chance" if new_ci[1] < 0.5 else "above chance"),
            "source": "recomputed from data/cache/p10/old_test_features_cache.pkl with simultaneous token resampling",
        }
        print(f"  old_test    | {m_name:<12} | LM-AUROC: {lm_val:.4f} | Old CI: [{old_ci[0]:.4f}, {old_ci[1]:.4f}] -> New CI: [{new_ci[0]:.4f}, {new_ci[1]:.4f}]")

    # -------------------------------------------------------------------------
    # ITEM 2: V3 and V6 three-way counts on test2-clean
    # -------------------------------------------------------------------------
    print("\n--- ITEM 2: V3 and V6 Three-Way Counts on test2-clean ---")
    v3_3way = t2_clean_json["V3"]["three_way_decision"]
    v6_3way = t2_clean_json["V6"]["three_way_decision"]

    v3_safe = v3_3way["n_safe_flagged"]
    v3_vuln = v3_3way["n_vuln_flagged"]
    v3_unsure = v3_3way["n_unsure"]
    v3_sum = v3_safe + v3_vuln + v3_unsure

    v6_safe = v6_3way["n_safe_flagged"]
    v6_vuln = v6_3way["n_vuln_flagged"]
    v6_unsure = v6_3way["n_unsure"]
    v6_sum = v6_safe + v6_vuln + v6_unsure

    item2_fixes = {
        "V3": {
            "n_safe": v3_safe,
            "n_vuln": v3_vuln,
            "n_unsure": v3_unsure,
            "sum": v3_sum,
            "sum_equals_N": bool(v3_sum == 1991),
            "precision": v3_3way["p_vuln_precision"],
            "npv": v3_3way["p_safe_npv"],
            "unsure_rate": v3_3way["unsure_rate"],
            "source": "results/p11/test2_clean_evaluation.json, key test2_clean_norm.V3.three_way_decision",
        },
        "V6": {
            "old_reported_counts": {
                "n_safe": 810,
                "n_vuln": 20,
                "n_unsure": 1406,
                "sum": 2236,
                "error": "Prior draft reported counts from test2_clean_exact (N=2236) instead of test2_clean_norm (N=1991)",
            },
            "new_verified_counts": {
                "n_safe": v6_safe,
                "n_vuln": v6_vuln,
                "n_unsure": v6_unsure,
                "sum": v6_sum,
                "sum_equals_N": bool(v6_sum == 1991),
                "precision": v6_3way["p_vuln_precision"],
                "npv": v6_3way["p_safe_npv"],
                "unsure_rate": v6_3way["unsure_rate"],
            },
            "source": "results/p11/test2_clean_evaluation.json, key test2_clean_norm.V6.three_way_decision",
        },
    }
    print(f"  V3: safe={v3_safe}, vuln={v3_vuln}, unsure={v3_unsure} -> Sum = {v3_sum} (Precision: {v3_3way['p_vuln_precision']*100:.2f}%, NPV: {v3_3way['p_safe_npv']*100:.2f}%)")
    print(f"  V6: safe={v6_safe}, vuln={v6_vuln}, unsure={v6_unsure} -> Sum = {v6_sum} (Precision: {v6_3way['p_vuln_precision']*100:.2f}%, NPV: {v6_3way['p_safe_npv']*100:.2f}%)")

    # -------------------------------------------------------------------------
    # ITEM 3: Actual test2 and test2-clean project lists, N and prevalence
    # -------------------------------------------------------------------------
    print("\n--- ITEM 3: Project Lists, N, and Prevalence ---")
    with open(BASE_DIR / "results" / "p10" / "test2_projects.json") as f:
        t2_proj_json = json.load(f)

    t2_full_projects = t2_proj_json["project_breakdown"]
    df_clean = df_test2[mask_clean_norm]

    clean_proj_breakdown = {}
    for p_name, grp in df_clean.groupby("project"):
        clean_proj_breakdown[p_name] = {
            "n": int(len(grp)),
            "n_vuln": int(grp["vuln_label"].sum()),
            "prevalence": float(grp["vuln_label"].mean()),
        }

    item3_fixes = {
        "test2_full": {
            "N": int(len(df_test2)),
            "n_vuln": int(df_test2["vuln_label"].sum()),
            "prevalence": float(df_test2["vuln_label"].mean()),
            "projects_count": len(t2_full_projects),
            "cpp_projects": t2_proj_json["selected_summary"]["cpp_projects"],
            "c_projects": t2_proj_json["selected_summary"]["c_projects"],
            "project_breakdown": t2_full_projects,
            "source": "results/p10/test2_projects.json",
        },
        "test2_clean": {
            "N": int(len(df_clean)),
            "n_vuln": int(df_clean["vuln_label"].sum()),
            "prevalence": float(df_clean["vuln_label"].mean()),
            "projects_count": len(clean_proj_breakdown),
            "project_breakdown": clean_proj_breakdown,
            "source": "data/p10/test2.parquet filtered by normalized duplicate removal against train/dev/old_test",
        },
        "note_on_prior_hallucination": "HANDOFF_P10.md Section 3.1 hallucinated projects (git, redis, curl, systemd, libvirt, wireshark, qemu, php-src, nginx, openssl, envoy, vlc) with prevalence 18.04%. Verified ground truth test2 has 12 projects with prevalence 32.64% (full) and 20.84% (clean).",
    }
    print(f"  test2 full:  N={len(df_test2)}, Prevalence={df_test2['vuln_label'].mean()*100:.2f}%, Projects={list(t2_full_projects.keys())}")
    print(f"  test2-clean: N={len(df_clean)}, Prevalence={df_clean['vuln_label'].mean()*100:.2f}%, Projects={list(clean_proj_breakdown.keys())}")

    # -------------------------------------------------------------------------
    # ITEM 4: Per-project test2-clean metrics from saved predictions
    # -------------------------------------------------------------------------
    print("\n--- ITEM 4: Per-Project test2-clean Metrics ---")
    per_project_metrics = {}
    for proj, grp in df_clean.groupby("project"):
        p_idx = grp.index
        # Map to position in mask_clean_norm
        pos_in_clean = np.where(df_clean.index.isin(p_idx))[0]
        y_p = y_t2_clean[pos_in_clean]
        comm_p = commits_t2_clean[pos_in_clean]
        n_p = len(y_p)
        pos_count = int(np.sum(y_p))
        prev_p = float(np.mean(y_p))

        proj_dict = {
            "n": n_p,
            "positives": pos_count,
            "prevalence": prev_p,
            "models": {},
        }

        for m_name, preds_clean in models_t2_clean.items():
            sub_preds = preds_clean[pos_in_clean]
            if len(np.unique(y_p)) > 1:
                auc = float(roc_auc_score(y_p, sub_preds))
                auc_ci = list(cluster_bootstrap_auroc(y_p, sub_preds, comm_p, n_resamples=500, seed=42))
            else:
                auc = None
                auc_ci = None
            proj_dict["models"][m_name] = {
                "auroc": auc,
                "auroc_ci": auc_ci,
            }
        per_project_metrics[proj] = proj_dict

    item4_fixes = {
        "status_in_saved_p11_json": "UNAVAILABLE",
        "saved_artifact_note": "In results/p11/test2_clean_evaluation.json, per-project breakdown metrics were not saved (only dataset-level aggregates were stored).",
        "newly_computed_from_cache_and_metadata": per_project_metrics,
        "source": "Computed from data/cache/p10/test2_features_cache.pkl and data/p10/test2.parquet metadata",
    }
    for pr, pdat in per_project_metrics.items():
        v3_auc_str = f"{pdat['models']['V3']['auroc']:.4f}" if pdat['models']['V3']['auroc'] is not None else "UNDEF (0 pos)"
        print(f"  {pr:<14}: n={pdat['n']:<4}, positives={pdat['positives']:<3}, prev={pdat['prevalence']*100:5.2f}% | V3 AUROC: {v3_auc_str}")

    # -------------------------------------------------------------------------
    # ITEM 5: Old test N and prevalence
    # -------------------------------------------------------------------------
    print("\n--- ITEM 5: Old Test N and Prevalence ---")
    old_test_n = int(len(df_old_test))
    old_test_vuln = int(df_old_test["vuln_label"].sum())
    old_test_prev = float(df_old_test["vuln_label"].mean())
    old_test_projects = df_old_test["project"].value_counts().to_dict()

    item5_fixes = {
        "old_erroneous_claim": {
            "N": 4016,
            "prevalence": 0.3175,
            "source_error": "Prior drafts in reports/FINAL_RESULTS.md and COMPREHENSIVE_PROJECT_DOSSIER.md stated N=4,016 with prevalence 31.75%.",
        },
        "new_verified_value": {
            "N": old_test_n,
            "n_positives": old_test_vuln,
            "prevalence": old_test_prev,
            "projects": old_test_projects,
            "source": "results/p11/old_test_evaluation.json and data/splits/vulnerability_test_heldout_projects_clean.parquet",
        },
    }
    print(f"  Old test: N={old_test_n} (was 4,016), Prevalence={old_test_prev*100:.2f}% ({old_test_vuln}/{old_test_n}), Projects={old_test_projects}")

    # -------------------------------------------------------------------------
    # ITEM 6: Juliet Evaluation & Leak Audit
    # -------------------------------------------------------------------------
    print("\n--- ITEM 6: Juliet Splits and Leak Audit ---")
    with open(BASE_DIR / "results" / "p10" / "juliet_evaluation.json") as f:
        juliet_eval_json = json.load(f)
    with open(BASE_DIR / "results" / "p10" / "juliet_leak_check.json") as f:
        juliet_leak_json = json.load(f)

    juliet_splits_summary = {}
    for s_name, s_data in juliet_eval_json["splits"].items():
        leak_meta = juliet_leak_json["leak_check_splits"].get(s_name, {})
        split_dict = {
            "n_train": leak_meta.get("n_train"),
            "n_val": leak_meta.get("n_val"),
            "n_test": leak_meta.get("n_test"),
            "models": {},
        }
        for m_name, m_data in s_data.items():
            met = m_data.get("metrics", {})
            auc = met.get("auroc")
            acc = met.get("accuracy")
            pair_acc = m_data.get("pair_accuracy")
            ci = m_data.get("auroc_ci")
            is_leak = bool((auc is not None and auc > 0.98) or (acc is not None and acc > 0.98))
            split_dict["models"][m_name] = {
                "auroc": auc,
                "auroc_ci": ci,
                "accuracy": acc,
                "pair_accuracy": pair_acc,
                "possible_leak": is_leak,
            }
        juliet_splits_summary[s_name] = split_dict

    item6_fixes = {
        "old_erroneous_claim": {
            "Juliet_S_flow_N": 64099,
            "Juliet_S_flow_AUROC": 0.9997,
            "Juliet_S_cwe_N": 64099,
            "Juliet_S_cwe_AUROC": 0.6120,
            "source_error": "Prior FINAL_RESULTS.md table pasted fabricated N=64,099 and AUROC=0.9997.",
        },
        "verified_juliet_splits": juliet_splits_summary,
        "leak_audit_conclusion": "In results/p10/juliet_evaluation.json, true test splits are: split_flow (N=6,702, J2 AUROC=0.6925), split_cwe (N=5,441, J2 AUROC=0.7872), split_rand (N=2,736, J2 AUROC=0.8135). Any prior claim of AUROC > 0.98 on Juliet (such as 0.9997) is flagged as a synthetic memorization artifact or test-leak.",
        "source": "results/p10/juliet_evaluation.json and results/p10/juliet_leak_check.json",
    }
    for s_name, s_dat in juliet_splits_summary.items():
        j2_auc = s_dat["models"]["J2"]["auroc"]
        print(f"  {s_name:<12}: N_test={s_dat['n_test']}, J2 AUROC={j2_auc:.4f}")

    # -------------------------------------------------------------------------
    # ITEM 7: Model B Re-Audit & Verification
    # -------------------------------------------------------------------------
    print("\n--- ITEM 7: Model B Verification ---")
    with open(BASE_DIR / "data" / "p10" / "harnesses" / "harnesses.json") as f:
        harnesses_json = json.load(f)
    valid_harnesses = [k for k, v in harnesses_json.items() if v.get("status") == "valid"]

    with open(BASE_DIR / "results" / "p11" / "modelb_v2_labels.json") as f:
        mb_v2_json = json.load(f)

    with open(BASE_DIR / "data" / "p10" / "model_b_exec_labels.json") as f:
        el_raw = json.load(f)

    roles_count = {}
    for u in el_raw.values():
        r = u.get("role", "unknown")
        roles_count[r] = roles_count.get(r, 0) + 1

    item7_fixes = {
        "tasks_with_validated_harness": {
            "total_tasks_in_harnesses_json": len(harnesses_json),
            "validated_harness_count": len(valid_harnesses),
            "no_harness_count": len(harnesses_json) - len(valid_harnesses),
            "source": "data/p10/harnesses/harnesses.json",
        },
        "units_per_role": roles_count,
        "clean_tasks_count": mb_v2_json["c1_human_vs_ai_v2"]["n_clean_tasks"],
        "suspect_tasks_excluded": mb_v2_json["c1_human_vs_ai_v2"]["suspect_tasks_excluded"],
        "cross_validation_method_actual": "5-fold GroupKFold cross-validation grouped by task_id across 5 random seeds (42, 43, 44, 45, 46). Leave-One-Out (LOO) was NOT used.",
        "c1_counts_and_mcnemar": {
            "n_tasks": mb_v2_json["c1_human_vs_ai_v2"]["n_clean_tasks"],
            "rate_human": mb_v2_json["c1_human_vs_ai_v2"]["rate_human"],
            "rate_human_ci": mb_v2_json["c1_human_vs_ai_v2"]["rate_human_ci"],
            "rate_ai_t02": mb_v2_json["c1_human_vs_ai_v2"]["rate_ai_t02"],
            "rate_ai_t02_ci": mb_v2_json["c1_human_vs_ai_v2"]["rate_ai_t02_ci"],
            "diff_t02_minus_human": mb_v2_json["c1_human_vs_ai_v2"]["diff_t02_minus_human"],
            "mcnemar_t02": mb_v2_json["c1_human_vs_ai_v2"]["mcnemar_t02"],
            "rate_ai_t08": mb_v2_json["c1_human_vs_ai_v2"]["rate_ai_t08"],
            "rate_ai_t08_ci": mb_v2_json["c1_human_vs_ai_v2"]["rate_ai_t08_ci"],
            "diff_t08_minus_human": mb_v2_json["c1_human_vs_ai_v2"]["diff_t08_minus_human"],
            "mcnemar_t08": mb_v2_json["c1_human_vs_ai_v2"]["mcnemar_t08"],
        },
        "c2_aurocs_with_ci": mb_v2_json["c2_model_b_v2_clean"],
        "source": "results/p11/modelb_v2_labels.json and scripts/p10/model_b_pipeline.py",
    }
    print(f"  Valid harnesses: {len(valid_harnesses)} / {len(harnesses_json)}")
    print(f"  Clean tasks: {mb_v2_json['c1_human_vs_ai_v2']['n_clean_tasks']} (excluded 9 suspect tasks)")
    print(f"  C1 Human crash rate: {mb_v2_json['c1_human_vs_ai_v2']['rate_human']*100:.1f}%, AI t0.2: {mb_v2_json['c1_human_vs_ai_v2']['rate_ai_t02']*100:.1f}%, AI t0.8: {mb_v2_json['c1_human_vs_ai_v2']['rate_ai_t08']*100:.1f}%")

    # -------------------------------------------------------------------------
    # WRITE MASTER FIXES JSON
    # -------------------------------------------------------------------------
    fixes_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "item_1_length_matched_auroc_ci": item1_fixes,
        "item_2_threeway_counts": item2_fixes,
        "item_3_project_lists": item3_fixes,
        "item_4_per_project_test2_clean": item4_fixes,
        "item_5_old_test": item5_fixes,
        "item_6_juliet": item6_fixes,
        "item_7_model_b": item7_fixes,
    }

    out_path = RESULTS_P12_DIR / "fixes.json"
    with open(out_path, "w") as f:
        json.dump(fixes_json, f, indent=2)
    print(f"\n[Complete] Successfully wrote fixes to {out_path}")


if __name__ == "__main__":
    main()

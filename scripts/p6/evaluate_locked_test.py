"""P6 Single Final Evaluation of Frozen Model A on Locked Test Set.

Protocol:
1. Verifies results/p6/freeze.json exists.
2. Calls load_locked_test(reason="P6 single final evaluation of frozen Model A") exactly once.
3. Evaluates L1, Champion, and D2_ensemble at top 10% alert operating threshold.
4. Computes metrics on Full test and Deduplicated test (overall and per project)
   with 2,000 cluster bootstrap resamples by commit.
5. Extracts test pairs and computes pair_correct + operating budget partition (P-C, P-V, P-B, P-R).
6. Computes per-CWE recall for CWE-476, 400, 835, 416, 787, 119 with Wilson 95% CIs.
7. Computes paired hypothesis tests (diff, 95% cluster CI, raw p, Holm p).
8. Saves test predictions to results/p6/test_preds.parquet.
9. Saves complete results to results/p6/locked_test_results.json.
"""

import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer

from src.eval.locked_test import load_locked_test
from src.eval.stats import (
    wilson_ci,
    calc_length_stratified_auroc,
    holm_adjust,
    format_p_value,
)
from src.eval.stats_v2 import (
    calc_lsauroc10,
    get_length_matched_pair_indices,
    calc_length_matched_auroc,
    alert_budget_metrics,
    paired_metrics_at_cutoff,
    paired_bootstrap_diff_v2,
)
from scripts.p4.step1_paired_data import extract_disjoint_pairs
from models.checkpoints.p5.model_a_final.predictor import predict as model_a_predict

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
P2_SPLITS_DIR = DATA_DIR / "splits" / "p2"
RESULTS_P2_DIR = PROJECT_ROOT / "results" / "p2"
RESULTS_P5_DIR = PROJECT_ROOT / "results" / "p5"
RESULTS_P6_DIR = PROJECT_ROOT / "results" / "p6"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"

RESULTS_P6_DIR.mkdir(parents=True, exist_ok=True)


def compute_cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    metric_fn: Any,
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> List[float]:
    """Computes 95% CI via cluster bootstrap grouped by commit."""
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}

    extra_kwargs = {k: np.asarray(v) for k, v in extra_arrays.items()} if extra_arrays is not None else {}

    boot = []
    for _ in range(n_resamples):
        chosen_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        b_idx = np.concatenate([group_to_idx[g] for g in chosen_groups])
        y_sub = y_true[b_idx]
        if len(np.unique(y_sub)) < 2:
            continue
        p_sub = y_prob[b_idx]
        sub_extra = {k: v[b_idx] for k, v in extra_kwargs.items()} if extra_kwargs else {}
        try:
            val = float(metric_fn(y_sub, p_sub, **sub_extra))
            boot.append(val)
        except TypeError:
            try:
                val = float(metric_fn(y_sub, p_sub))
                boot.append(val)
            except Exception:
                continue
        except Exception:
            continue

    if len(boot) == 0:
        return [0.0, 1.0]
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def evaluate_cohort(
    cohort_name: str,
    mask: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    toks: np.ndarray,
    commits: np.ndarray,
    q10_edges: np.ndarray,
    matched_v: np.ndarray,
    matched_s: np.ndarray,
    n_resamples: int = 2000,
) -> Dict[str, Any]:
    """Evaluates n, prevalence, AUROC, LS-AUROC 4-bin, 10-bin, length-matched, AUPRC, alert top 10%."""
    y_c = y[mask]
    p_c = p[mask]
    toks_c = toks[mask]
    commits_c = commits[mask]

    n = len(y_c)
    n_pos = int(np.sum(y_c == 1))
    prevalence = float(n_pos / n) if n > 0 else 0.0

    if n_pos == 0 or n_pos == n:
        return {
            "n": n, "n_pos": n_pos, "prevalence": prevalence,
            "auroc": 0.5, "auroc_ci_95": [0.5, 0.5],
            "ls_auroc_4bin": 0.5, "ls_auroc_4bin_ci_95": [0.5, 0.5],
            "ls_auroc_10bin": 0.5, "ls_auroc_10bin_ci_95": [0.5, 0.5],
            "length_matched_auroc": 0.5, "length_matched_ci_95": [0.5, 0.5], "matched_pairs_count": 0,
            "auprc": prevalence, "auprc_ci_95": [prevalence, prevalence],
            "alert_top10": {"precision": 0.0, "recall": 0.0, "lift": 0.0, "threshold": 0.5},
        }

    # AUROC
    auc = float(roc_auc_score(y_c, p_c))
    ci_auc = compute_cluster_bootstrap_ci(y_c, p_c, commits_c, roc_auc_score, n_resamples=n_resamples)

    # 4-bin LS-AUROC
    ls4 = float(calc_length_stratified_auroc(y_c, p_c, toks_c))
    ci_ls4 = compute_cluster_bootstrap_ci(
        y_c, p_c, commits_c, calc_length_stratified_auroc, extra_arrays={"n_tokens": toks_c}, n_resamples=n_resamples
    )

    # 10-bin LS-AUROC
    ls10 = float(calc_lsauroc10(y_c, p_c, toks_c, quantile_edges=q10_edges))
    def _ls10_metric(y_sub, p_sub, **kw):
        return calc_lsauroc10(y_sub, p_sub, kw["n_tokens"], quantile_edges=q10_edges)
    ci_ls10 = compute_cluster_bootstrap_ci(
        y_c, p_c, commits_c, _ls10_metric, extra_arrays={"n_tokens": toks_c}, n_resamples=n_resamples
    )

    # Length-matched AUROC (|dlog toks| <= 0.1) within cohort
    v_m, s_m = get_length_matched_pair_indices(y_c, toks_c, max_log_diff=0.1)
    n_pairs = len(v_m)
    if n_pairs > 0:
        lm_auc = float(calc_length_matched_auroc(y_c, p_c, toks_c, pair_indices=(v_m, s_m)))
        def _lm_metric(y_sub, p_sub, **kw):
            return calc_length_matched_auroc(y_sub, p_sub, n_tokens=kw["n_tokens"], max_log_diff=0.1)
        ci_lm = compute_cluster_bootstrap_ci(
            y_c, p_c, commits_c, _lm_metric, extra_arrays={"n_tokens": toks_c}, n_resamples=n_resamples
        )
    else:
        lm_auc = 0.5
        ci_lm = [0.5, 0.5]

    # AUPRC
    auprc = float(average_precision_score(y_c, p_c))
    ci_auprc = compute_cluster_bootstrap_ci(y_c, p_c, commits_c, average_precision_score, n_resamples=n_resamples)

    # Alert top 10% within this cohort
    alert = alert_budget_metrics(y_c, p_c, budget_fraction=0.10)

    return {
        "n": n,
        "n_pos": n_pos,
        "prevalence": prevalence,
        "auroc": auc,
        "auroc_ci_95": ci_auc,
        "ls_auroc_4bin": ls4,
        "ls_auroc_4bin_ci_95": ci_ls4,
        "ls_auroc_10bin": ls10,
        "ls_auroc_10bin_ci_95": ci_ls10,
        "length_matched_auroc": lm_auc,
        "length_matched_ci_95": ci_lm,
        "matched_pairs_count": n_pairs,
        "auprc": auprc,
        "auprc_ci_95": ci_auprc,
        "alert_top10": {
            "precision": alert["precision"],
            "recall": alert["recall"],
            "lift": alert["lift"],
            "threshold": alert["operating_threshold"],
        },
    }


def main():
    t_start = time.time()
    print("=" * 80)
    print("PHASE 16 - PROMPT 6: FROZEN MODEL A EVALUATION ON LOCKED TEST SET")
    print("=" * 80)

    # 1. Verify Freeze Record exists
    freeze_path = RESULTS_P6_DIR / "freeze.json"
    if not freeze_path.exists():
        raise RuntimeError(f"Freeze record missing! {freeze_path} must exist before touching test.")
    with open(freeze_path, "r") as f:
        freeze_rec = json.load(f)
    print(f"[1] Verified freeze record {freeze_path} (recorded at {freeze_rec['timestamp_utc']}, {freeze_rec['file_count']} files).")

    # 2. Load Locked Test Set (CALLED ONCE)
    print("\n[2] Loading locked test set via load_locked_test()...")
    df_test = load_locked_test(reason="P6 single final evaluation of frozen Model A")
    print(f"Loaded locked test parquet: {len(df_test)} samples")

    y_test = df_test["vuln_label"].values.astype(int)
    projects_test = df_test["project"].str.lower().values
    commits_test = df_test["commit_id"].values

    # Compute CodeBERT token lengths for test samples
    print("Tokenizing test samples with CodeBERT tokenizer for exact length...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    tok_test = np.array([
        len(tokenizer.encode(str(c), add_special_tokens=False, truncation=False))
        for c in df_test["code"]
    ])
    print(f"Computed test token lengths: min={tok_test.min()}, max={tok_test.max()}, median={np.median(tok_test):.0f}")

    # 3. Model 1: L1 refit on train_minus_dev
    print("\n[3] Fitting L1 Baseline on train_minus_dev...")
    df_train = pd.read_parquet(P2_SPLITS_DIR / "train_minus_dev.parquet")
    tok_train = np.load(CACHE_DIR / "train_token_lens.npy")

    sc_l1 = StandardScaler()
    clf_l1 = LogisticRegression(C=0.01, class_weight="balanced", random_state=42)
    clf_l1.fit(sc_l1.fit_transform(np.log1p(tok_train).reshape(-1, 1)), df_train["vuln_label"].values)
    score_l1 = clf_l1.predict_proba(sc_l1.transform(np.log1p(tok_test).reshape(-1, 1)))[:, 1]
    print(f"L1 test scores computed (mean={score_l1.mean():.4f})")

    # 4. Model 2: Phase 15 Champion (faithful C=0.01 recipe)
    print("\n[4] Computing Champion test predictions from cached features...")
    with open(MODELS_DIR / "phase15_codebert_cxx_hybrid_best.pkl", "rb") as f:
        ckpt_champ = pickle.load(f)

    # Load precomputed test features from cache
    emb_champ = np.load(CACHE_DIR / "emb_codebert_256_ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc.npy")
    cxx28_test = np.load(CACHE_DIR / "feat_cxx28_ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc.npy")
    X_champ = np.hstack([emb_champ, ckpt_champ["scaler"].transform(cxx28_test)])
    score_champion = ckpt_champ["clf"].predict_proba(X_champ)[:, 1]
    print(f"Champion test scores computed (mean={score_champion.mean():.4f})")

    # 5. Model 3: D2_ensemble
    print("\n[5] Running D2_ensemble predict() on test snippets...")
    t_inf_start = time.time()
    score_d2_ensemble = model_a_predict(df_test["code"].tolist(), batch_size=32)
    print(f"D2_ensemble inference completed in {time.time() - t_inf_start:.2f}s (mean={score_d2_ensemble.mean():.4f})")

    # 6. Save Predictions Parquet
    preds_df = pd.DataFrame({
        "id": df_test["sample_id"],
        "project": df_test["project"],
        "cwe": df_test["cwe"],
        "n_tokens": tok_test,
        "label": y_test,
        "score_l1": score_l1,
        "score_champion": score_champion,
        "score_d2_ensemble": score_d2_ensemble,
    })
    preds_parquet_path = RESULTS_P6_DIR / "test_preds.parquet"
    preds_df.to_parquet(preds_parquet_path, index=False)
    print(f"\n[6] Saved test predictions to {preds_parquet_path} ({len(preds_df)} rows)")

    # 7. Define Cohorts and Deduplication
    print("\n[7] Defining Evaluation Cohorts (Full vs Deduplicated)...")
    # Identify near-duplicate sample IDs from P2 audit
    df_near_dups = pd.read_csv(RESULTS_P2_DIR / "near_dups_test_train.csv")
    test_dup_ids = set(df_near_dups[df_near_dups["jaccard"] >= 0.8]["test_sample_id"])
    is_non_dup = ~df_test["sample_id"].isin(test_dup_ids).values
    print(f"Near-duplicate test samples identified: {len(test_dup_ids)}")
    print(f"Deduplicated test sample count: {int(np.sum(is_non_dup))}")

    # Envoy production code: non-mock
    gtest_regex = re.compile(r'\b(EXPECT_CALL|ON_CALL|TEST_F|TEST_P|TEST|MOCK_METHOD|ASSERT_TRUE|ASSERT_FALSE|EXPECT_TRUE|EXPECT_FALSE|EXPECT_EQ|ASSERT_EQ)\b')
    is_mock = np.array([bool(gtest_regex.search(str(c))) for c in df_test["code"]])
    is_envoy = (projects_test == "envoy")
    is_envoy_prod = is_envoy & (~is_mock)

    cohort_masks_full = {
        "Overall": np.ones(len(df_test), dtype=bool),
        "Envoy": is_envoy,
        "Envoy Production": is_envoy_prod,
        "OpenSSL": (projects_test == "openssl"),
        "Vim": (projects_test == "vim"),
        "GPAC": (projects_test == "gpac"),
    }

    cohort_masks_dedup = {
        name: mask & is_non_dup for name, mask in cohort_masks_full.items()
    }

    # Load 10-quantile bin edges from dev (as established in P5)
    with open(RESULTS_P5_DIR / "step1_evaluation_fixes.json", "r") as f:
        step1_res = json.load(f)
    q10_edges = np.array(step1_res["A3_quantile_binning_sensitivity"]["q10_edges"])
    print(f"Loaded 10-quantile bin edges from dev: {[round(x, 1) for x in q10_edges]}")

    # 8. Full and Deduplicated Evaluations for Steps 4
    print("\n[8] Evaluating Full and Deduplicated Test Sets (2000 Cluster Bootstrap Resamples)...")
    models_dict = {
        "L1": score_l1,
        "Champion": score_champion,
        "D2_ensemble": score_d2_ensemble,
    }

    eval_results_full = {}
    eval_results_dedup = {}

    for m_name, m_prob in models_dict.items():
        print(f"\n--- Model: {m_name} ---")
        eval_results_full[m_name] = {}
        for c_name, c_mask in cohort_masks_full.items():
            print(f"  Full Test - {c_name}...")
            eval_results_full[m_name][c_name] = evaluate_cohort(
                c_name, c_mask, y_test, m_prob, tok_test, commits_test, q10_edges, None, None, n_resamples=2000
            )

        eval_results_dedup[m_name] = {}
        for c_name, c_mask in cohort_masks_dedup.items():
            print(f"  Deduplicated Test - {c_name}...")
            eval_results_dedup[m_name][c_name] = evaluate_cohort(
                c_name, c_mask, y_test, m_prob, tok_test, commits_test, q10_edges, None, None, n_resamples=2000
            )

    # 9. Test Pairs Evaluation (Step 5)
    print("\n[9] Extracting Test Pairs and Computing Paired Operating Metrics...")
    test_pairs = extract_disjoint_pairs(df_test)
    n_test_pairs = len(test_pairs)
    print(f"Extracted {n_test_pairs} disjoint test pairs.")

    test_id_to_idx = {sid: i for i, sid in enumerate(df_test["sample_id"])}
    v_pair_idx = np.array([test_id_to_idx[p["vuln_sample_id"]] for p in test_pairs])
    f_pair_idx = np.array([test_id_to_idx[p["fixed_sample_id"]] for p in test_pairs])

    # For each model, compute operating threshold at top 10% on full test
    pairs_evaluation = {}
    rng_pairs = np.random.RandomState(42)

    for m_name, m_prob in models_dict.items():
        tau_10 = eval_results_full[m_name]["Overall"]["alert_top10"]["threshold"]
        p_v = m_prob[v_pair_idx]
        p_f = m_prob[f_pair_idx]

        pair_m = paired_metrics_at_cutoff(p_v, p_f, tau=tau_10)

        # 95% bootstrap CI for pair_correct (2000 resamples)
        corr_vec = (p_v > p_f).astype(float) + 0.5 * (p_v == p_f).astype(float)
        boot_corr = []
        for _ in range(2000):
            idx = rng_pairs.choice(n_test_pairs, size=n_test_pairs, replace=True)
            boot_corr.append(np.mean(corr_vec[idx]))
        ci_corr = [float(np.percentile(boot_corr, 2.5)), float(np.percentile(boot_corr, 97.5))]

        pairs_evaluation[m_name] = {
            "pair_count": n_test_pairs,
            "operating_threshold": tau_10,
            "pair_correct": pair_m["pair_correct"],
            "pair_correct_ci_95": ci_corr,
            "P-C": pair_m["P-C"],
            "P-V": pair_m["P-V"],
            "P-B": pair_m["P-B"],
            "P-R": pair_m["P-R"],
        }
        print(f"  {m_name}: pair_correct={pair_m['pair_correct']*100:.1f}% [{ci_corr[0]*100:.1f}%, {ci_corr[1]*100:.1f}%], P-C={pair_m['P-C']*100:.1f}%, P-V={pair_m['P-V']*100:.1f}%, P-B={pair_m['P-B']*100:.1f}%, P-R={pair_m['P-R']*100:.1f}%")

    # 10. Per-CWE Recall at 10% Budget (Step 6)
    print("\n[10] Computing Per-CWE Recall at 10% Alert Operating Budget...")
    TARGET_CWES = ["CWE-476", "CWE-400", "CWE-835", "CWE-416", "CWE-787", "CWE-119"]
    cwe_test = df_test["cwe"].fillna("").astype(str).values

    cwe_evaluation = {}
    for cwe in TARGET_CWES:
        cwe_mask = (y_test == 1) & (cwe_test.str.contains(cwe) if hasattr(cwe_test, 'str') else np.array([cwe in s for s in cwe_test]))
        n_cwe = int(np.sum(cwe_mask))
        cwe_evaluation[cwe] = {"n_total": n_cwe, "models": {}}

        for m_name, m_prob in models_dict.items():
            tau_10 = eval_results_full[m_name]["Overall"]["alert_top10"]["threshold"]
            if n_cwe > 0:
                k_flagged = int(np.sum(m_prob[cwe_mask] >= tau_10))
                recall = float(k_flagged / n_cwe)
                low_w, high_w = wilson_ci(k_flagged, n_cwe, 0.95)
            else:
                k_flagged = 0
                recall = 0.0
                low_w, high_w = 0.0, 0.0

            cwe_evaluation[cwe]["models"][m_name] = {
                "k_flagged": k_flagged,
                "n_total": n_cwe,
                "recall": recall,
                "wilson_ci_95": [low_w, high_w],
            }
            print(f"  {cwe} | {m_name}: {k_flagged}/{n_cwe} ({recall*100:.1f}%) [Wilson: {low_w*100:.1f}%, {high_w*100:.1f}%]")

    # 11. Paired Tests D2_ensemble vs L1 and vs Champion (Step 7)
    print("\n[11] Computing Paired Hypothesis Tests (D2_ensemble vs L1, vs Champion)...")
    def _lm_fn(y_sub, p_sub, **kw):
        return calc_length_matched_auroc(y_sub, p_sub, n_tokens=kw["n_tokens"], max_log_diff=0.1)

    paired_tests_output = {}
    for comp_name, comp_prob in [("L1", score_l1), ("Champion", score_champion)]:
        # AUROC
        diff_auc = paired_bootstrap_diff_v2(
            y_test, comp_prob, score_d2_ensemble,
            metric_fn=roc_auc_score,
            groups=commits_test,
            n_resamples=2000,
            seed=42,
        )
        # LS-AUROC 4-bin
        diff_ls4 = paired_bootstrap_diff_v2(
            y_test, comp_prob, score_d2_ensemble,
            metric_fn=calc_length_stratified_auroc,
            groups=commits_test,
            extra_arrays={"n_tokens": tok_test},
            n_resamples=2000,
            seed=42,
        )
        # Length-Matched AUROC
        diff_lm = paired_bootstrap_diff_v2(
            y_test, comp_prob, score_d2_ensemble,
            metric_fn=_lm_fn,
            groups=commits_test,
            extra_arrays={"n_tokens": tok_test},
            n_resamples=2000,
            seed=42,
        )

        paired_tests_output[f"D2_ensemble_vs_{comp_name}"] = {
            "auroc": diff_auc,
            "ls_auroc_4bin": diff_ls4,
            "length_matched_auroc": diff_lm,
        }

    # Holm correction across the 6 tests (3 metrics x 2 comparisons)
    all_raw_p = [
        paired_tests_output["D2_ensemble_vs_L1"]["auroc"]["raw_p_value"],
        paired_tests_output["D2_ensemble_vs_L1"]["ls_auroc_4bin"]["raw_p_value"],
        paired_tests_output["D2_ensemble_vs_L1"]["length_matched_auroc"]["raw_p_value"],
        paired_tests_output["D2_ensemble_vs_Champion"]["auroc"]["raw_p_value"],
        paired_tests_output["D2_ensemble_vs_Champion"]["ls_auroc_4bin"]["raw_p_value"],
        paired_tests_output["D2_ensemble_vs_Champion"]["length_matched_auroc"]["raw_p_value"],
    ]
    all_holm_p = holm_adjust(all_raw_p)

    test_keys = [
        ("D2_ensemble_vs_L1", "auroc"),
        ("D2_ensemble_vs_L1", "ls_auroc_4bin"),
        ("D2_ensemble_vs_L1", "length_matched_auroc"),
        ("D2_ensemble_vs_Champion", "auroc"),
        ("D2_ensemble_vs_Champion", "ls_auroc_4bin"),
        ("D2_ensemble_vs_Champion", "length_matched_auroc"),
    ]

    for idx, (c_key, m_key) in enumerate(test_keys):
        paired_tests_output[c_key][m_key]["p_value_holm"] = all_holm_p[idx]
        paired_tests_output[c_key][m_key]["p_value_holm_formatted"] = format_p_value(all_holm_p[idx], 0.0005)

    # 12. Save Complete Results JSON
    full_output = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_runtime_seconds": float(time.time() - t_start),
        "test_parquet": {
            "path": "data/splits/vulnerability_test_heldout_projects_clean.parquet",
            "sample_count": len(df_test),
            "deduplicated_sample_count": int(np.sum(is_non_dup)),
        },
        "full_test_evaluation": eval_results_full,
        "deduplicated_test_evaluation": eval_results_dedup,
        "pairs_evaluation": pairs_evaluation,
        "cwe_evaluation": cwe_evaluation,
        "paired_tests": paired_tests_output,
        "l1_pair_correct_explanation": "In P4 ties scored 0.0 (strict wins: 17.7%), while in P5 ties were credited 0.5 (17.7% + 0.5*7.0% ties = 21.2%).",
    }

    out_json = RESULTS_P6_DIR / "locked_test_results.json"
    with open(out_json, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nSaved locked test results to {out_json}")

    print("\n" + "=" * 80)
    print(f"LOCKED TEST EVALUATION COMPLETED IN {time.time() - t_start:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()

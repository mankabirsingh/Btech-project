"""Step 4: Evaluation, Ensembling, and Mechanical Model Selection.

Implements:
- D1: Per-seed evaluation on dev_heldout and dev_pairs.
- Multi-seed summary: mean +/- std for D2 (5 seeds), PAIR (3 seeds), N0 (2 seeds).
- D2: Recipe ensembles (mean of logits across seeds).
- D3: Paired comparisons (grouped by commit, Holm-adjusted across rows) vs L1, vs champion, and PAIR vs D2.
- D4: Mechanical model selection following results/p5/preregistration.json.

Saves:
- results/p5/p5_master_evaluation.json
- results/p5/model_selection_decision.json
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import scipy.special
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from scripts.p3.eval_protocol import get_cached_features, load_datasets
from src.eval.stats import (
    calc_length_stratified_auroc,
    bootstrap_ci,
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

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
P2_SPLITS_DIR = SPLITS_DIR / "p2"
P4_SPLITS_DIR = SPLITS_DIR / "p4"
P5_SPLITS_DIR = SPLITS_DIR / "p5"
RESULTS_P4_DIR = PROJECT_ROOT / "results" / "p4"
RESULTS_P5_DIR = PROJECT_ROOT / "results" / "p5"


def compute_pair_correct_with_ci(
    p_vuln: np.ndarray,
    p_fixed: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    n = len(p_vuln)
    is_correct = (p_vuln > p_fixed).astype(float) + 0.5 * (p_vuln == p_fixed).astype(float)
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot.append(np.mean(is_correct[idx]))
    return {
        "score": float(np.mean(is_correct)),
        "ci_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def compute_candidate_metrics(
    p_cand: np.ndarray,
    y_dev: np.ndarray,
    tok_dev: np.ndarray,
    is_cpp: np.ndarray,
    dev_projects: np.ndarray,
    matched_v_idx: np.ndarray,
    matched_s_idx: np.ndarray,
    v_pair_idx: np.ndarray,
    f_pair_idx: np.ndarray,
    q10_edges: np.ndarray,
) -> Dict[str, Any]:
    """Computes all core evaluation metrics for a single prediction array on dev."""
    # AUROC
    auc_val = float(roc_auc_score(y_dev, p_cand))
    ci_auc = bootstrap_ci(y_dev, p_cand, n_resamples=2000, seed=42)["auroc_ci_95"]

    # 4-bin LS-AUROC
    ls4_val = float(calc_length_stratified_auroc(y_dev, p_cand, tok_dev))
    rng = np.random.RandomState(42)
    pos_idx = np.where(y_dev == 1)[0]
    neg_idx = np.where(y_dev == 0)[0]
    boot_ls4 = []
    for _ in range(2000):
        b_idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True)
        ])
        boot_ls4.append(calc_length_stratified_auroc(y_dev[b_idx], p_cand[b_idx], tok_dev[b_idx]))
    ci_ls4 = [float(np.percentile(boot_ls4, 2.5)), float(np.percentile(boot_ls4, 97.5))]

    # 10-bin LS-AUROC
    ls10_val = float(calc_lsauroc10(y_dev, p_cand, tok_dev, quantile_edges=q10_edges))
    boot_ls10 = []
    for _ in range(2000):
        b_idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True)
        ])
        boot_ls10.append(calc_lsauroc10(y_dev[b_idx], p_cand[b_idx], tok_dev[b_idx], quantile_edges=q10_edges))
    ci_ls10 = [float(np.percentile(boot_ls10, 2.5)), float(np.percentile(boot_ls10, 97.5))]

    # Length-matched AUROC (46,251 pairs)
    lm_auc_val = float(calc_length_matched_auroc(y_dev, p_cand, tok_dev, pair_indices=(matched_v_idx, matched_s_idx)))
    p_v_match = p_cand[matched_v_idx]
    p_s_match = p_cand[matched_s_idx]
    boot_lm = []
    n_mp = len(matched_v_idx)
    for _ in range(2000):
        b_idx = rng.choice(n_mp, size=n_mp, replace=True)
        wins = np.sum(p_v_match[b_idx] > p_s_match[b_idx])
        ties = np.sum(p_v_match[b_idx] == p_s_match[b_idx])
        boot_lm.append((wins + 0.5 * ties) / n_mp)
    ci_lm = [float(np.percentile(boot_lm, 2.5)), float(np.percentile(boot_lm, 97.5))]

    # C++ AUROC
    cpp_auc = float(roc_auc_score(y_dev[is_cpp], p_cand[is_cpp])) if np.sum(is_cpp) > 0 else 0.5

    # Dev pairs evaluation (186 pairs)
    p_vuln_pairs = p_cand[v_pair_idx]
    p_fixed_pairs = p_cand[f_pair_idx]
    pair_corr_dict = compute_pair_correct_with_ci(p_vuln_pairs, p_fixed_pairs, n_resamples=2000, seed=42)

    # Alert budget top 10%
    alert_overall = alert_budget_metrics(y_dev, p_cand, budget_fraction=0.10)
    tau_10 = alert_overall["operating_threshold"]

    per_project_alert = {}
    for proj in ["qpdf", "openexr", "tcpdump", "radare2"]:
        mask_p = (dev_projects == proj)
        if np.sum(mask_p) > 0:
            p_res = alert_budget_metrics(y_dev[mask_p], p_cand[mask_p], budget_fraction=0.10)
            per_project_alert[proj] = {
                "n_total": p_res["n_total"],
                "prevalence": p_res["prevalence"],
                "precision": p_res["precision"],
                "recall": p_res["recall"],
                "lift": p_res["lift"],
            }

    paired_alert = paired_metrics_at_cutoff(p_vuln_pairs, p_fixed_pairs, tau=tau_10)

    return {
        "dev_auroc": auc_val,
        "dev_auroc_ci_95": ci_auc,
        "dev_ls_auroc": ls4_val,
        "dev_ls_auroc_ci_95": ci_ls4,
        "dev_lsauroc10": ls10_val,
        "dev_lsauroc10_ci_95": ci_ls10,
        "length_matched_auroc": lm_auc_val,
        "length_matched_ci_95": ci_lm,
        "dev_cpp_auroc": cpp_auc,
        "pair_correct": pair_corr_dict["score"],
        "pair_correct_ci_95": pair_corr_dict["ci_95"],
        "operating_threshold_top10": tau_10,
        "alert_overall": {
            "precision": alert_overall["precision"],
            "recall": alert_overall["recall"],
            "lift": alert_overall["lift"],
            "prevalence": alert_overall["prevalence"],
        },
        "alert_per_project": per_project_alert,
        "paired_alert_dev_pairs": {
            "P-C": paired_alert["P-C"],
            "P-V": paired_alert["P-V"],
            "P-B": paired_alert["P-B"],
            "P-R": paired_alert["P-R"],
        },
    }


def main():
    t_start = time.time()
    print("=" * 80)
    print("STEP 4: EVALUATION, ENSEMBLING, AND MECHANICAL MODEL SELECTION")
    print("=" * 80)

    # 1. Load data
    df_train, df_dev, tok_train, tok_dev = load_datasets()
    df_pairs = pd.read_parquet(P4_SPLITS_DIR / "dev_pairs.parquet")

    y_dev = df_dev["vuln_label"].values.astype(int)
    dev_commits = df_dev["commit_id"].values
    dev_projects = df_dev["project"].str.lower().values
    is_cpp = df_dev["is_cpp"].astype(bool).values

    dev_id_to_idx = {sid: i for i, sid in enumerate(df_dev["sample_id"])}
    v_pair_idx = np.array([dev_id_to_idx[sid] for sid in df_pairs["vuln_sample_id"]])
    f_pair_idx = np.array([dev_id_to_idx[sid] for sid in df_pairs["fixed_sample_id"]])

    matched_v_idx, matched_s_idx = get_length_matched_pair_indices(y_dev, tok_dev, max_log_diff=0.1)

    q10_edges = np.quantile(tok_dev, np.linspace(0, 1, 11))
    q10_edges[0] = 0.0
    q10_edges[-1] = np.inf

    # 2. Reference Baselines: L1 and Faithful Champion P2
    # L1
    sc_l1 = StandardScaler()
    clf_l1 = LogisticRegression(C=0.01, class_weight="balanced", random_state=42)
    clf_l1.fit(sc_l1.fit_transform(np.log1p(tok_train).reshape(-1, 1)), df_train["vuln_label"])
    p_l1 = clf_l1.predict_proba(sc_l1.transform(np.log1p(tok_dev).reshape(-1, 1)))[:, 1]

    # Faithful Champion P2
    feats = get_cached_features()
    sc_cxx = StandardScaler()
    sc_cxx_tr = sc_cxx.fit_transform(feats["train"]["cxx28"])
    sc_cxx_dv = sc_cxx.transform(feats["dev"]["cxx28"])
    clf_champ = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=42)
    clf_champ.fit(np.hstack([feats["train"]["emb_codebert_256"], sc_cxx_tr]), df_train["vuln_label"])
    p_champ = clf_champ.predict_proba(np.hstack([feats["dev"]["emb_codebert_256"], sc_cxx_dv]))[:, 1]

    metrics_l1 = compute_candidate_metrics(
        p_l1, y_dev, tok_dev, is_cpp, dev_projects, matched_v_idx, matched_s_idx, v_pair_idx, f_pair_idx, q10_edges
    )
    metrics_champ = compute_candidate_metrics(
        p_champ, y_dev, tok_dev, is_cpp, dev_projects, matched_v_idx, matched_s_idx, v_pair_idx, f_pair_idx, q10_edges
    )

    # 3. Load all per-seed predictions and logits
    seeds_d2 = [42, 7, 1, 2, 3]
    seeds_pair = [42, 7, 1]
    seeds_n0 = [42, 7]

    per_seed_results = {}
    logits_d2_list = []
    logits_pair_list = []
    logits_n0_list = []

    # D2 seeds
    for s in seeds_d2:
        p_file = RESULTS_P5_DIR / f"p_dev_D2_seed{s}_max.npy"
        l_file = RESULTS_P5_DIR / f"logits_dev_D2_seed{s}_max.npy"
        if not p_file.exists():
            raise FileNotFoundError(f"Missing prediction file: {p_file}")
        p_s = np.load(p_file)
        l_s = np.load(l_file) if l_file.exists() else scipy.special.logit(np.clip(p_s, 1e-6, 1.0 - 1e-6))
        logits_d2_list.append(l_s)
        m_s = compute_candidate_metrics(
            p_s, y_dev, tok_dev, is_cpp, dev_projects, matched_v_idx, matched_s_idx, v_pair_idx, f_pair_idx, q10_edges
        )
        per_seed_results[f"D2_seed{s}"] = m_s

    # PAIR seeds
    for s in seeds_pair:
        p_file = RESULTS_P5_DIR / f"p_dev_PAIR_seed{s}_max.npy"
        l_file = RESULTS_P5_DIR / f"logits_dev_PAIR_seed{s}_max.npy"
        if not p_file.exists():
            raise FileNotFoundError(f"Missing prediction file: {p_file}")
        p_s = np.load(p_file)
        l_s = np.load(l_file) if l_file.exists() else scipy.special.logit(np.clip(p_s, 1e-6, 1.0 - 1e-6))
        logits_pair_list.append(l_s)
        m_s = compute_candidate_metrics(
            p_s, y_dev, tok_dev, is_cpp, dev_projects, matched_v_idx, matched_s_idx, v_pair_idx, f_pair_idx, q10_edges
        )
        per_seed_results[f"PAIR_seed{s}"] = m_s

    # N0 seeds
    for s in seeds_n0:
        p_file = RESULTS_P5_DIR / f"p_dev_N0_seed{s}_max.npy"
        l_file = RESULTS_P5_DIR / f"logits_dev_N0_seed{s}_max.npy"
        if not p_file.exists():
            raise FileNotFoundError(f"Missing prediction file: {p_file}")
        p_s = np.load(p_file)
        l_s = np.load(l_file) if l_file.exists() else scipy.special.logit(np.clip(p_s, 1e-6, 1.0 - 1e-6))
        logits_n0_list.append(l_s)
        m_s = compute_candidate_metrics(
            p_s, y_dev, tok_dev, is_cpp, dev_projects, matched_v_idx, matched_s_idx, v_pair_idx, f_pair_idx, q10_edges
        )
        per_seed_results[f"N0_seed{s}"] = m_s

    # 4. Multi-seed summary (mean +/- std)
    def calc_mean_std(model_prefix: str, seeds: List[int]) -> Dict[str, Any]:
        metrics = ["dev_auroc", "dev_ls_auroc", "dev_lsauroc10", "length_matched_auroc", "dev_cpp_auroc", "pair_correct"]
        summary = {}
        for m in metrics:
            vals = [per_seed_results[f"{model_prefix}_seed{s}"][m] for s in seeds]
            summary[m] = {
                "values": vals,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        return summary

    multi_seed_summary = {
        "D2_5seeds": calc_mean_std("D2", seeds_d2),
        "PAIR_3seeds": calc_mean_std("PAIR", seeds_pair),
        "N0_2seeds": calc_mean_std("N0", seeds_n0),
    }

    # 5. Form Recipe Ensembles (Arithmetic average of logits)
    ens_logits_d2 = np.mean(logits_d2_list, axis=0)
    ens_p_d2 = scipy.special.expit(ens_logits_d2)

    ens_logits_pair = np.mean(logits_pair_list, axis=0)
    ens_p_pair = scipy.special.expit(ens_logits_pair)

    ens_logits_n0 = np.mean(logits_n0_list, axis=0)
    ens_p_n0 = scipy.special.expit(ens_logits_n0)

    # Save ensemble predictions
    np.save(RESULTS_P5_DIR / "p_dev_D2_ensemble.npy", ens_p_d2)
    np.save(RESULTS_P5_DIR / "p_dev_PAIR_ensemble.npy", ens_p_pair)
    np.save(RESULTS_P5_DIR / "p_dev_N0_ensemble.npy", ens_p_n0)

    ensemble_metrics = {
        "D2_ensemble": compute_candidate_metrics(
            ens_p_d2, y_dev, tok_dev, is_cpp, dev_projects, matched_v_idx, matched_s_idx, v_pair_idx, f_pair_idx, q10_edges
        ),
        "PAIR_ensemble": compute_candidate_metrics(
            ens_p_pair, y_dev, tok_dev, is_cpp, dev_projects, matched_v_idx, matched_s_idx, v_pair_idx, f_pair_idx, q10_edges
        ),
        "N0_ensemble": compute_candidate_metrics(
            ens_p_n0, y_dev, tok_dev, is_cpp, dev_projects, matched_v_idx, matched_s_idx, v_pair_idx, f_pair_idx, q10_edges
        ),
    }

    # 6. Paired hypothesis tests (2000-resample cluster bootstrap by commit)
    def _lm_fn(y, p, **kw):
        return calc_length_matched_auroc(y, p, n_tokens=kw["n_tokens"], max_log_diff=0.1)

    ensembles_dict = {
        "D2_ensemble": ens_p_d2,
        "PAIR_ensemble": ens_p_pair,
        "N0_ensemble": ens_p_n0,
    }

    paired_tests = {}
    for name, p_ens in ensembles_dict.items():
        # vs L1 on LS-AUROC
        diff_ls_l1 = paired_bootstrap_diff_v2(
            y_dev, p_l1, p_ens,
            metric_fn=calc_length_stratified_auroc,
            groups=dev_commits,
            extra_arrays={"n_tokens": tok_dev},
            n_resamples=2000,
            seed=42,
        )
        # vs L1 on Length-Matched AUROC
        diff_lm_l1 = paired_bootstrap_diff_v2(
            y_dev, p_l1, p_ens,
            metric_fn=_lm_fn,
            groups=dev_commits,
            extra_arrays={"n_tokens": tok_dev},
            n_resamples=2000,
            seed=42,
        )
        # vs Champion on LS-AUROC
        diff_ls_champ = paired_bootstrap_diff_v2(
            y_dev, p_champ, p_ens,
            metric_fn=calc_length_stratified_auroc,
            groups=dev_commits,
            extra_arrays={"n_tokens": tok_dev},
            n_resamples=2000,
            seed=42,
        )
        # vs Champion on Length-Matched AUROC
        diff_lm_champ = paired_bootstrap_diff_v2(
            y_dev, p_champ, p_ens,
            metric_fn=_lm_fn,
            groups=dev_commits,
            extra_arrays={"n_tokens": tok_dev},
            n_resamples=2000,
            seed=42,
        )

        paired_tests[name] = {
            "vs_L1": {
                "ls_auroc": diff_ls_l1,
                "length_matched_auroc": diff_lm_l1,
            },
            "vs_champion": {
                "ls_auroc": diff_ls_champ,
                "length_matched_auroc": diff_lm_champ,
            },
        }

    # Holm adjust p-values across the 3 ensembles
    ens_names = list(ensembles_dict.keys())
    for comp in ["vs_L1", "vs_champion"]:
        for metric in ["ls_auroc", "length_matched_auroc"]:
            raw_p = [paired_tests[k][comp][metric]["raw_p_value"] for k in ens_names]
            adj_p = holm_adjust(raw_p)
            for i, k in enumerate(ens_names):
                paired_tests[k][comp][metric]["p_value_holm"] = adj_p[i]
                paired_tests[k][comp][metric]["p_value_holm_formatted"] = format_p_value(adj_p[i], 0.0005)

    # Paired test: PAIR ensemble vs D2 ensemble
    diff_pair_vs_d2_ls = paired_bootstrap_diff_v2(
        y_dev, ens_p_d2, ens_p_pair,
        metric_fn=calc_length_stratified_auroc,
        groups=dev_commits,
        extra_arrays={"n_tokens": tok_dev},
        n_resamples=2000,
        seed=42,
    )
    diff_pair_vs_d2_lm = paired_bootstrap_diff_v2(
        y_dev, ens_p_d2, ens_p_pair,
        metric_fn=_lm_fn,
        groups=dev_commits,
        extra_arrays={"n_tokens": tok_dev},
        n_resamples=2000,
        seed=42,
    )
    # Paired test on dev_pairs (pair_correct)
    p_vuln_d2 = ens_p_d2[v_pair_idx]
    p_fixed_d2 = ens_p_d2[f_pair_idx]
    p_vuln_pair = ens_p_pair[v_pair_idx]
    p_fixed_pair = ens_p_pair[f_pair_idx]

    corr_d2 = (p_vuln_d2 > p_fixed_d2).astype(float) + 0.5 * (p_vuln_d2 == p_fixed_d2).astype(float)
    corr_pair = (p_vuln_pair > p_fixed_pair).astype(float) + 0.5 * (p_vuln_pair == p_fixed_pair).astype(float)
    diff_corr_obs = float(np.mean(corr_pair) - np.mean(corr_d2))

    rng = np.random.RandomState(42)
    n_pairs = len(v_pair_idx)
    boot_diff_corr = []
    for _ in range(2000):
        b_idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        boot_diff_corr.append(np.mean(corr_pair[b_idx]) - np.mean(corr_d2[b_idx]))
    ci_lower_corr = float(np.percentile(boot_diff_corr, 2.5))
    ci_upper_corr = float(np.percentile(boot_diff_corr, 97.5))
    p_val_corr = float(min(1.0, 2.0 * min(np.mean(np.array(boot_diff_corr) <= 0), np.mean(np.array(boot_diff_corr) >= 0))))

    pair_vs_d2_tests = {
        "ls_auroc": diff_pair_vs_d2_ls,
        "length_matched_auroc": diff_pair_vs_d2_lm,
        "pair_correct": {
            "score_d2": float(np.mean(corr_d2)),
            "score_pair": float(np.mean(corr_pair)),
            "diff_observed": diff_corr_obs,
            "ci_lower": ci_lower_corr,
            "ci_upper": ci_upper_corr,
            "raw_p_value": p_val_corr,
            "p_value_formatted": format_p_value(p_val_corr, 0.0005),
        },
    }

    # 7. Apply Pre-Registered Mechanical Model Selection Rule
    with open(RESULTS_P5_DIR / "preregistration.json", "r") as f:
        prereg = json.load(f)

    decision_log = []
    eligible_candidates = []

    for cand_name in ["D2_ensemble", "PAIR_ensemble", "N0_ensemble"]:
        m = ensemble_metrics[cand_name]
        t = paired_tests[cand_name]

        # Gate 1: Beats L1 on length_matched_auroc with Holm p < 0.05
        gate1_p = t["vs_L1"]["length_matched_auroc"]["p_value_holm"]
        gate1_pass = bool(gate1_p < 0.05 and t["vs_L1"]["length_matched_auroc"]["diff_observed"] > 0)

        # Gate 2: Dev LS-AUROC not significantly worse than champion
        gate2_diff = t["vs_champion"]["ls_auroc"]["diff_observed"]
        gate2_p = t["vs_champion"]["ls_auroc"]["raw_p_value"]
        gate2_pass = bool(gate2_diff >= 0 or gate2_p >= 0.05)

        composite_score = 0.5 * (m["length_matched_auroc"] + m["pair_correct"])

        cand_status = {
            "candidate": cand_name,
            "gate_1": {
                "metric": "length_matched_auroc vs L1",
                "diff": t["vs_L1"]["length_matched_auroc"]["diff_observed"],
                "p_holm": gate1_p,
                "passed": gate1_pass,
            },
            "gate_2": {
                "metric": "dev_ls_auroc vs champion",
                "diff": gate2_diff,
                "p_raw": gate2_p,
                "passed": gate2_pass,
            },
            "length_matched_auroc": m["length_matched_auroc"],
            "length_matched_ci_95": m["length_matched_ci_95"],
            "pair_correct": m["pair_correct"],
            "pair_correct_ci_95": m["pair_correct_ci_95"],
            "composite_score": composite_score,
            "eligible": bool(gate1_pass and gate2_pass),
        }
        decision_log.append(cand_status)
        if cand_status["eligible"]:
            eligible_candidates.append(cand_status)

    # Optimization & Tie breaking
    eligible_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    if not eligible_candidates:
        raise RuntimeError("No candidate passed Gates 1 and 2!")

    winner_name = eligible_candidates[0]["candidate"]
    winning_reason = f"Ranked #1 by composite score ({eligible_candidates[0]['composite_score']:.4f})."

    # Check tiebreaker if both PAIR and D2 are eligible
    pair_status = next((c for c in eligible_candidates if c["candidate"] == "PAIR_ensemble"), None)
    d2_status = next((c for c in eligible_candidates if c["candidate"] == "D2_ensemble"), None)

    if pair_status and d2_status:
        # Check if composite scores overlap within 95% CI
        diff_lm = pair_vs_d2_tests["length_matched_auroc"]["diff_observed"]
        ci_lm_low = pair_vs_d2_tests["length_matched_auroc"]["ci_lower"]
        ci_lm_high = pair_vs_d2_tests["length_matched_auroc"]["ci_upper"]
        p_lm = pair_vs_d2_tests["length_matched_auroc"]["raw_p_value"]

        # If difference between PAIR and D2 on length-matched AUROC contains 0 (statistically indistinguishable)
        if ci_lm_low <= 0.0 <= ci_lm_high:
            winner_name = "D2_ensemble"
            winning_reason = (
                f"Tie-breaker triggered: PAIR_ensemble and D2_ensemble are statistically indistinguishable "
                f"on length_matched_auroc (Delta={diff_lm:+.4f}, 95% CI [{ci_lm_low:+.4f}, {ci_lm_high:+.4f}], p={p_lm:.4f}). "
                f"Pre-registered parsimony rule selects D2_ensemble."
            )
        else:
            winner_name = eligible_candidates[0]["candidate"]
            winning_reason = f"Significantly higher length_matched_auroc with non-overlapping CI."

    selection_decision = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "winning_candidate": winner_name,
        "winning_reason": winning_reason,
        "decision_log": decision_log,
        "pair_vs_d2_comparison": pair_vs_d2_tests,
    }

    # Save outputs
    master_eval = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "runtime_seconds": float(time.time() - t_start),
        "baselines": {
            "L1": metrics_l1,
            "faithful_champion_p2": metrics_champ,
        },
        "per_seed_results": per_seed_results,
        "multi_seed_summary": multi_seed_summary,
        "ensemble_metrics": ensemble_metrics,
        "paired_tests": paired_tests,
        "selection_decision": selection_decision,
    }

    with open(RESULTS_P5_DIR / "p5_master_evaluation.json", "w") as f:
        json.dump(master_eval, f, indent=2)
    print(f"Saved master evaluation to {RESULTS_P5_DIR / 'p5_master_evaluation.json'}")

    with open(RESULTS_P5_DIR / "model_selection_decision.json", "w") as f:
        json.dump(selection_decision, f, indent=2)
    print(f"Saved selection decision to {RESULTS_P5_DIR / 'model_selection_decision.json'}")

    print("\n" + "=" * 80)
    print(f"WINNING MODEL SELECTED: {winner_name}")
    print(f"REASON: {winning_reason}")
    print("=" * 80)


if __name__ == "__main__":
    main()

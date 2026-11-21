"""STEP 1: Confound Baselines and Length Prevalence Analysis.

Trains logistic regression baselines L1, L2, L3, L4, and partials out length from champion_p2.
Also reports prevalence by token length bin across train_minus_dev, dev_heldout, and test_clean.
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.p3.eval_protocol import (
    evaluate_candidate,
    get_cached_features,
    get_champion_p2_reference,
    load_datasets,
)
from src.eval.stats import paired_bootstrap_diff

RESULTS_P3_DIR = PROJECT_ROOT / "results" / "p3"
RESULTS_P3_DIR.mkdir(parents=True, exist_ok=True)


def extract_cyclomatic_proxy(codes: pd.Series) -> np.ndarray:
    """Count of if, for, while, case, catch, &&, || in code."""
    pattern = re.compile(r"\b(if|for|while|case|catch)\b|&&|\|\|")
    counts = [len(pattern.findall(str(c))) for c in codes]
    return np.array(counts, dtype=np.float32)


def main():
    print("=" * 80)
    print("STEP 1: CONFOUND BASELINES & LENGTH PREVALENCE ANALYSIS")
    print("=" * 80)
    t_start = time.time()

    df_train, df_dev, tok_train, tok_dev = load_datasets()
    features = get_cached_features()
    champ_ref = get_champion_p2_reference(df_train, df_dev, tok_dev, features)

    print(f"Loaded {len(df_train):,} train samples and {len(df_dev):,} dev samples.")

    # 1. Length Prevalence Breakdown
    print("\n[1] Computing Prevalence by Length Bins...")
    bins = [0, 256, 512, 1024, np.inf]
    bin_labels = ["<= 256", "257 - 512", "513 - 1024", "> 1024"]

    def compute_bin_stats(y: np.ndarray, toks: np.ndarray) -> Dict[str, Any]:
        stats = {}
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            if i == 0:
                mask = (toks >= low) & (toks <= high)
            else:
                mask = (toks > low) & (toks <= high)
            n_b = int(np.sum(mask))
            n_pos = int(np.sum(y[mask] == 1))
            prev = float(n_pos / n_b) if n_b > 0 else 0.0
            stats[bin_labels[i]] = {
                "n": n_b,
                "n_vuln": n_pos,
                "prevalence": prev,
            }
        return stats

    y_train = df_train["vuln_label"].values.astype(int)
    y_dev = df_dev["vuln_label"].values.astype(int)

    train_len_stats = compute_bin_stats(y_train, tok_train)
    dev_len_stats = compute_bin_stats(y_dev, tok_dev)

    # Test numbers from locked test benchmark known from Phase 16 P2 audit
    test_len_stats = {
        "<= 256": {"n": 1373, "n_vuln": 199, "prevalence": 199 / 1373},
        "257 - 512": {"n": 558, "n_vuln": 153, "prevalence": 153 / 558},
        "513 - 1024": {"n": 454, "n_vuln": 153, "prevalence": 153 / 454},
        "> 1024": {"n": 784, "n_vuln": 498, "prevalence": 498 / 784},
    }

    len_prevalence_table = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "description": "Prevalence by CodeBERT token length bin across splits",
        },
        "train_minus_dev": train_len_stats,
        "dev_heldout": dev_len_stats,
        "test_heldout_clean": test_len_stats,
    }

    len_json_path = RESULTS_P3_DIR / "step1_length_prevalence.json"
    with open(len_json_path, "w") as f:
        json.dump(len_prevalence_table, f, indent=2)

    print("\n--- Length Bin Prevalence Table ---")
    print(f"{'Bin':<14} | {'Train N':<8} {'Train Prev':<11} | {'Dev N':<6} {'Dev Prev':<9} | {'Test N':<6} {'Test Prev':<10}")
    print("-" * 75)
    for b in bin_labels:
        tr = train_len_stats[b]
        dv = dev_len_stats[b]
        ts = test_len_stats[b]
        print(f"{b:<14} | {tr['n']:<8} {tr['prevalence']*100:>6.2f}%    | {dv['n']:<6} {dv['prevalence']*100:>5.2f}%   | {ts['n']:<6} {ts['prevalence']*100:>5.2f}%")

    # 2. Extract Confound Features
    print("\n[2] Extracting Confound Features...")
    log_tok_train = np.log1p(tok_train).reshape(-1, 1).astype(np.float32)
    log_tok_dev = np.log1p(tok_dev).reshape(-1, 1).astype(np.float32)

    lines_train = np.array([str(c).count("\n") + 1 for c in df_train["code"]], dtype=np.float32)
    lines_dev = np.array([str(c).count("\n") + 1 for c in df_dev["code"]], dtype=np.float32)
    log_lines_train = np.log1p(lines_train).reshape(-1, 1)
    log_lines_dev = np.log1p(lines_dev).reshape(-1, 1)

    cyc_train = np.log1p(extract_cyclomatic_proxy(df_train["code"])).reshape(-1, 1)
    cyc_dev = np.log1p(extract_cyclomatic_proxy(df_dev["code"])).reshape(-1, 1)

    X_L1_train = log_tok_train
    X_L1_dev = log_tok_dev

    X_L2_train = np.hstack([log_tok_train, log_lines_train, cyc_train])
    X_L2_dev = np.hstack([log_tok_dev, log_lines_dev, cyc_dev])

    X_L3_train = features["train"]["cxx28"]
    X_L3_dev = features["dev"]["cxx28"]

    X_L4_train = features["train"]["cxx38"]
    X_L4_dev = features["dev"]["cxx38"]

    # 3. Train & Evaluate Baselines L1, L2, L3, L4
    candidates = [
        ("L1", "Logistic Regression on log(n_tokens) only", "Confound Baseline", X_L1_train, X_L1_dev, 0.01),
        ("L2", "Logistic Regression on log(n_tokens) + log(n_lines) + cyclomatic proxy", "Confound Baseline", X_L2_train, X_L2_dev, 0.01),
        ("L3", "Logistic Regression on cxx28 features only", "Confound Baseline", X_L3_train, X_L3_dev, 0.01),
        ("L4", "Logistic Regression on cxx38 features only", "Confound Baseline", X_L4_train, X_L4_dev, 0.01),
    ]

    results_baselines = {}
    for cid, desc, cat, x_tr, x_dv, c_val in candidates:
        print(f"\n--- Evaluating {cid}: {desc} ---")
        res = evaluate_candidate(
            candidate_id=cid,
            description=desc,
            category=cat,
            X_train_raw=x_tr,
            X_dev_raw=x_dv,
            df_train=df_train,
            df_dev=df_dev,
            tok_dev=tok_dev,
            champ_ref=champ_ref,
            C_val=c_val,
        )
        print(f"  Dev AUROC: {res['dev_auroc']:.4f} {res['dev_auroc_ci_95']}")
        print(f"  Dev Length-Stratified AUROC: {res['dev_length_stratified_auroc']:.4f} {res['dev_length_stratified_ci_95']}")
        print(f"  Dev C++ AUROC: {res['dev_cpp_auroc']:.4f} {res['dev_cpp_auroc_ci_95']}")
        print(f"  CV Macro AUROC: {res['cv_mean_auroc']:.4f} +/- {res['cv_std_auroc']:.4f}")
        results_baselines[cid] = res

    # 4. Partialling out length from Champion P2
    # Logistic regression of label on [log(n_tokens), champion_p2 logit]
    print("\n--- Evaluating Champion P2 Score Partialling Out Length ---")
    p_champ_train_oof = np.array(champ_ref["p_oof"])
    p_champ_dev = np.array(champ_ref["p_dev"])

    eps = 1e-6
    logit_train = np.log(np.clip(p_champ_train_oof, eps, 1 - eps) / (1 - np.clip(p_champ_train_oof, eps, 1 - eps))).reshape(-1, 1)
    logit_dev = np.log(np.clip(p_champ_dev, eps, 1 - eps) / (1 - np.clip(p_champ_dev, eps, 1 - eps))).reshape(-1, 1)

    X_partial_train = np.hstack([log_tok_train, logit_train])
    X_partial_dev = np.hstack([log_tok_dev, logit_dev])

    res_partial = evaluate_candidate(
        candidate_id="L_champion_partial",
        description="Logistic regression on [log(n_tokens), champion_p2 logit]",
        category="Confound Baseline",
        X_train_raw=X_partial_train,
        X_dev_raw=X_partial_dev,
        df_train=df_train,
        df_dev=df_dev,
        tok_dev=tok_dev,
        champ_ref=champ_ref,
        C_val=0.01,
    )

    # Gain in AUROC over L1
    l1_auc = results_baselines["L1"]["dev_auroc"]
    part_auc = res_partial["dev_auroc"]
    gain_over_l1 = float(part_auc - l1_auc)

    # Paired test vs L1
    def _auc_fn(y_sub, p_sub, _):
        return roc_auc_score(y_sub, p_sub) if len(np.unique(y_sub)) > 1 else 0.5

    paired_vs_l1 = paired_bootstrap_diff(
        y_dev,
        np.array(results_baselines["L1"]["p_dev"]),
        np.array(res_partial["p_dev"]),
        _auc_fn,
        groups=df_dev["commit_id"].values,
        n_resamples=1000,
        seed=42,
    )

    res_partial["gain_over_L1"] = gain_over_l1
    res_partial["paired_test_vs_L1"] = paired_vs_l1
    print(f"  Partial Model Dev AUROC: {part_auc:.4f} (L1 AUROC: {l1_auc:.4f}, Gain: {gain_over_l1:+.4f})")
    print(f"  Paired test vs L1: Diff = {paired_vs_l1['diff_observed']:+.4f} [{paired_vs_l1['ci_lower']:+.4f}, {paired_vs_l1['ci_upper']:+.4f}], p = {paired_vs_l1['p_value']:.4e}")

    results_baselines["L_champion_partial"] = res_partial

    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
        },
        "champion_p2_reference": {
            k: v for k, v in champ_ref.items() if k not in ["p_dev", "p_oof"]
        },
        "baselines": {
            k: {sk: sv for sk, sv in v.items() if sk not in ["p_dev", "p_oof"]}
            for k, v in results_baselines.items()
        },
    }

    out_json = RESULTS_P3_DIR / "step1_confounds.json"
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nStep 1 successfully saved to {out_json} in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

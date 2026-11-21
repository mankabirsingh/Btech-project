"""Step 4: Master Comparison and Mechanical Model Selection."""

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.p3.eval_protocol import load_datasets
from scripts.p4.step1_paired_data import compute_paired_metrics
from src.eval.stats import (
    bootstrap_ci,
    calc_length_stratified_auroc,
    format_p_value,
    holm_adjust,
    length_stratified_auroc,
    paired_bootstrap_diff,
)

P4_SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "p4"
P4_MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "p4"
RESULTS_P4_DIR = PROJECT_ROOT / "results" / "p4"
SELECTED_DIR = P4_MODELS_DIR / "model_a_p4_selected"


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("STEP 4: MASTER COMPARISON AND MECHANICAL MODEL SELECTION", flush=True)
    print("=" * 80, flush=True)

    # Load datasets
    df_train, df_dev, tok_train, tok_dev = load_datasets()
    df_pairs = pd.read_parquet(P4_SPLITS_DIR / "dev_pairs.parquet")

    y_dev = df_dev["vuln_label"].values
    dev_commits = df_dev["commit_id"].values
    is_cpp = df_dev["is_cpp"].astype(bool).values

    dev_id_to_idx = {sid: i for i, sid in enumerate(df_dev["sample_id"])}
    v_idx = [dev_id_to_idx[sid] for sid in df_pairs["vuln_sample_id"]]
    f_idx = [dev_id_to_idx[sid] for sid in df_pairs["fixed_sample_id"]]

    # Load Step 0 predictions & metadata
    with open(RESULTS_P4_DIR / "step0_resolutions.json", "r") as f:
        step0_res = json.load(f)
    draw_std_ls = step0_res["draw_variance_0_2"]["dev_ls_auroc_std"]

    with open(RESULTS_P4_DIR / "step0_p3_master_recomputed.json", "r") as f:
        step0_p3 = json.load(f)

    # Prediction dictionary for dev_heldout
    p_dev_dict = {}

    # 1. L1
    # Train L1 deterministically to get exact p_dev
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc_l1 = StandardScaler()
    clf_l1 = LogisticRegression(C=0.01, class_weight="balanced", random_state=42)
    clf_l1.fit(sc_l1.fit_transform(np.log1p(tok_train).reshape(-1, 1)), df_train["vuln_label"])
    p_dev_dict["L1"] = clf_l1.predict_proba(sc_l1.transform(np.log1p(tok_dev).reshape(-1, 1)))[:, 1]

    # 2. Faithful Champion P2
    from scripts.p3.eval_protocol import get_cached_features
    feats = get_cached_features()
    sc_cxx = StandardScaler()
    sc_cxx_tr = sc_cxx.fit_transform(feats["train"]["cxx28"])
    sc_cxx_dv = sc_cxx.transform(feats["dev"]["cxx28"])
    clf_champ = LogisticRegression(C=0.003, class_weight="balanced", max_iter=1000, random_state=42)
    clf_champ.fit(np.hstack([feats["train"]["emb_codebert_256"], sc_cxx_tr]), df_train["vuln_label"])
    p_dev_dict["faithful_champion_p2"] = clf_champ.predict_proba(np.hstack([feats["dev"]["emb_codebert_256"], sc_cxx_dv]))[:, 1]

    # 3. R2
    clf_r2 = LogisticRegression(C=0.003, class_weight="balanced", max_iter=1000, random_state=42)
    clf_r2.fit(np.hstack([feats["train"]["emb_chunked_mean"], sc_cxx_tr]), df_train["vuln_label"])
    p_dev_dict["R2"] = clf_r2.predict_proba(np.hstack([feats["dev"]["emb_chunked_mean"], sc_cxx_dv]))[:, 1]

    # 4. D1, D2, D3
    for cand in ["D1", "D2", "D3"]:
        p_path = RESULTS_P4_DIR / f"p_dev_{cand.lower()}.npy"
        if p_path.exists():
            p_dev_dict[cand] = np.load(p_path)

    # 5. F1, F2, F3 (UniXcoder fine-tuning)
    for run in ["F1", "F2", "F3"]:
        for pool in ["max", "mean"]:
            # Check for candidate files
            # Matches names like p_dev_f1_unixcoder_debiased_max.npy
            candidates_found = list(RESULTS_P4_DIR.glob(f"p_dev_{run.lower()}*_{pool}.npy"))
            if candidates_found:
                p_dev_dict[f"{run}_{pool}"] = np.load(candidates_found[0])

    print(f"Loaded dev predictions for {len(p_dev_dict)} models: {list(p_dev_dict.keys())}", flush=True)

    # Baseline references
    p_l1 = p_dev_dict["L1"]
    p_champ = p_dev_dict["faithful_champion_p2"]

    # Compute dev metrics for all candidates
    records = {}
    for cand_name, p_cand in p_dev_dict.items():
        dev_auc = float(roc_auc_score(y_dev, p_cand))
        dev_ls_auc = float(calc_length_stratified_auroc(y_dev, p_cand, tok_dev))
        cpp_auc = float(roc_auc_score(y_dev[is_cpp], p_cand[is_cpp])) if np.sum(is_cpp) > 0 else 0.5
        cpp_ls_auc = float(calc_length_stratified_auroc(y_dev[is_cpp], p_cand[is_cpp], tok_dev[is_cpp])) if np.sum(is_cpp) > 0 else 0.5
        paired_m = compute_paired_metrics(p_cand[v_idx], p_cand[f_idx], threshold=0.5, n_resamples=2000, seed=42)

        records[cand_name] = {
            "candidate": cand_name,
            "dev_auroc": dev_auc,
            "dev_length_stratified_auroc": dev_ls_auc,
            "dev_cpp_auroc": cpp_auc,
            "dev_cpp_length_stratified_auroc": cpp_ls_auc,
            "pair_correct": paired_m["pair_correct"],
            "pair_correct_ci_95": paired_m["pair_correct_ci_95"],
            "P-C": paired_m["P-C"],
            "P-V": paired_m["P-V"],
            "P-B": paired_m["P-B"],
            "P-R": paired_m["P-R"],
        }

    # Paired Bootstrap Differences vs L1 and vs faithful_champion_p2
    # Grouped by commit_id, 2000 resamples
    print("\nComputing paired bootstrap differences against L1 and faithful champion (2000 resamples)...", flush=True)
    comp_cands = [c for c in records.keys() if c != "L1"]

    raw_p_vs_l1_ls = []
    raw_p_vs_l1_ov = []
    raw_p_vs_champ_ls = []
    raw_p_vs_champ_ov = []

    diffs_vs_l1 = {}
    diffs_vs_champ = {}

    for cand_name in comp_cands:
        p_cand = p_dev_dict[cand_name]

        # Diff vs L1 on LS-AUROC
        diff_l1_ls = paired_bootstrap_diff(
            y_dev, p_l1, p_cand,
            lambda y, p, _, **extra: calc_length_stratified_auroc(y, p, extra["n_tokens"]),
            groups=dev_commits, extra_arrays={"n_tokens": tok_dev}, n_resamples=2000, seed=42,
        )
        diff_l1_ov = paired_bootstrap_diff(
            y_dev, p_l1, p_cand,
            lambda y, p, _: roc_auc_score(y, p),
            groups=dev_commits, n_resamples=2000, seed=42,
        )

        # Diff vs Faithful Champion on LS-AUROC & Overall
        diff_champ_ls = paired_bootstrap_diff(
            y_dev, p_champ, p_cand,
            lambda y, p, _, **extra: calc_length_stratified_auroc(y, p, extra["n_tokens"]),
            groups=dev_commits, extra_arrays={"n_tokens": tok_dev}, n_resamples=2000, seed=42,
        )
        diff_champ_ov = paired_bootstrap_diff(
            y_dev, p_champ, p_cand,
            lambda y, p, _: roc_auc_score(y, p),
            groups=dev_commits, n_resamples=2000, seed=42,
        )

        diffs_vs_l1[cand_name] = {"ls_auroc": diff_l1_ls, "overall_auroc": diff_l1_ov}
        diffs_vs_champ[cand_name] = {"ls_auroc": diff_champ_ls, "overall_auroc": diff_champ_ov}

        raw_p_vs_l1_ls.append(diff_l1_ls["p_value"])
        raw_p_vs_l1_ov.append(diff_l1_ov["p_value"])
        raw_p_vs_champ_ls.append(diff_champ_ls["p_value"])
        raw_p_vs_champ_ov.append(diff_champ_ov["p_value"])

    # Holm adjustments
    adj_p_l1_ls = holm_adjust(raw_p_vs_l1_ls)
    adj_p_l1_ov = holm_adjust(raw_p_vs_l1_ov)
    adj_p_champ_ls = holm_adjust(raw_p_vs_champ_ls)
    adj_p_champ_ov = holm_adjust(raw_p_vs_champ_ov)

    for idx, cand_name in enumerate(comp_cands):
        diffs_vs_l1[cand_name]["ls_auroc"]["p_value_holm"] = float(adj_p_l1_ls[idx])
        diffs_vs_l1[cand_name]["ls_auroc"]["p_value_holm_formatted"] = format_p_value(float(adj_p_l1_ls[idx]))
        diffs_vs_l1[cand_name]["overall_auroc"]["p_value_holm"] = float(adj_p_l1_ov[idx])
        diffs_vs_l1[cand_name]["overall_auroc"]["p_value_holm_formatted"] = format_p_value(float(adj_p_l1_ov[idx]))

        diffs_vs_champ[cand_name]["ls_auroc"]["p_value_holm"] = float(adj_p_champ_ls[idx])
        diffs_vs_champ[cand_name]["ls_auroc"]["p_value_holm_formatted"] = format_p_value(float(adj_p_champ_ls[idx]))
        diffs_vs_champ[cand_name]["overall_auroc"]["p_value_holm"] = float(adj_p_champ_ov[idx])
        diffs_vs_champ[cand_name]["overall_auroc"]["p_value_holm_formatted"] = format_p_value(float(adj_p_champ_ov[idx]))

        records[cand_name]["diff_vs_l1"] = diffs_vs_l1[cand_name]
        records[cand_name]["diff_vs_faithful_champion"] = diffs_vs_champ[cand_name]

    # Print Master Table
    print("\n" + "=" * 120, flush=True)
    print(f"{'Candidate':<24} | {'Dev AUROC':<10} | {'LS-AUROC':<10} | {'C++ AUROC':<10} | {'C++ LS-AUC':<10} | {'Pair Correct':<12} | {'Delta vs L1 (LS)':<16} | {'Holm p (L1)'}", flush=True)
    print("-" * 120, flush=True)

    l1_entry = records["L1"]
    print(f"{'L1':<24} | {l1_entry['dev_auroc']:<10.4f} | {l1_entry['dev_length_stratified_auroc']:<10.4f} | {l1_entry['dev_cpp_auroc']:<10.4f} | {l1_entry['dev_cpp_length_stratified_auroc']:<10.4f} | {l1_entry['pair_correct']*100:<11.2f}% | {'[Reference]':<16} | {'N/A'}", flush=True)

    for cand_name in comp_cands:
        r = records[cand_name]
        d_l1_ls = r["diff_vs_l1"]["ls_auroc"]["diff_observed"]
        p_l1_ls = r["diff_vs_l1"]["ls_auroc"]["p_value_holm_formatted"]
        print(f"{cand_name:<24} | {r['dev_auroc']:<10.4f} | {r['dev_length_stratified_auroc']:<10.4f} | {r['dev_cpp_auroc']:<10.4f} | {r['dev_cpp_length_stratified_auroc']:<10.4f} | {r['pair_correct']*100:<11.2f}% | {d_l1_ls:<+16.4f} | {p_l1_ls}", flush=True)
    print("=" * 120, flush=True)

    # -------------------------------------------------------------------------
    # MECHANICAL SELECTION RULE
    # -------------------------------------------------------------------------
    # "choose the candidate with the highest dev LS-AUROC among those whose LS-AUROC
    # beats L1 with Holm p < 0.05. If none beats L1, select nothing; report that
    # Model A does not exceed the length baseline and keep R2 as the working model."
    print("\n--- Applying Mechanical Selection Rule ---", flush=True)
    eligible_candidates = []
    for cand_name in comp_cands:
        r = records[cand_name]
        p_holm = r["diff_vs_l1"]["ls_auroc"]["p_value_holm"]
        diff_obs = r["diff_vs_l1"]["ls_auroc"]["diff_observed"]
        if diff_obs > 0 and p_holm < 0.05:
            eligible_candidates.append(cand_name)

    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    selected_meta = {}

    if eligible_candidates:
        selected_cand = max(eligible_candidates, key=lambda c: records[c]["dev_length_stratified_auroc"])
        print(f"Eligible candidates beating L1 with Holm p < 0.05: {eligible_candidates}", flush=True)
        print(f"SELECTED CANDIDATE: {selected_cand} (LS-AUROC = {records[selected_cand]['dev_length_stratified_auroc']:.4f})", flush=True)

        # Determine candidate metadata & copy weights
        data_variant = "D2 debiased (inverse-propensity weighted)" if any(k in selected_cand for k in ["F1", "F3", "D2"]) else ("Natural N0 control" if "F2" in selected_cand else selected_cand)
        run_name = selected_cand.split("_")[0] if "_" in selected_cand else selected_cand
        pooling = selected_cand.split("_")[1] if "_" in selected_cand else "N/A"

        tau = 0.5
        source_ckpt = None
        if selected_cand.startswith("F"):
            with open(RESULTS_P4_DIR / "step3_finetuning_eval.json", "r") as f:
                s3_eval = json.load(f)
            r_info = s3_eval.get(run_name, {})
            pool_key = "max_logit" if pooling == "max" else "mean_logit"
            tau = r_info.get("evaluation", {}).get(pool_key, {}).get("operating_threshold_10fpr", 0.5)
            subdirs = list(P4_MODELS_DIR.glob(f"{run_name}_*"))
            if subdirs and (subdirs[0] / "best_model.pt").exists():
                source_ckpt = subdirs[0] / "best_model.pt"
                shutil.copy2(source_ckpt, SELECTED_DIR / "best_model.pt")
                print(f"Copied {source_ckpt} to {SELECTED_DIR / 'best_model.pt'}", flush=True)
        elif selected_cand in ["D1", "D2", "D3"]:
            with open(RESULTS_P4_DIR / "step2_debiased_models.json", "r") as f:
                s2_eval = json.load(f)
            tau = s2_eval.get("eval_results", {}).get(selected_cand, {}).get("operating_threshold_10fpr", 0.5)
            p_cand_file = list(P4_MODELS_DIR.glob(f"{selected_cand.lower()}_*.pkl"))
            if p_cand_file:
                source_ckpt = p_cand_file[0]
                shutil.copy2(source_ckpt, SELECTED_DIR / "best_model.pkl")
                print(f"Copied {source_ckpt} to {SELECTED_DIR / 'best_model.pkl'}", flush=True)

        selected_meta = {
            "selected": True,
            "candidate": selected_cand,
            "data_variant": data_variant,
            "run": run_name,
            "inference_pooling": pooling,
            "operating_threshold_10fpr": float(tau),
            "source_checkpoint": str(source_ckpt) if source_ckpt else None,
            "dev_length_stratified_auroc": records[selected_cand]["dev_length_stratified_auroc"],
            "delta_vs_l1_ls_auroc": records[selected_cand]["diff_vs_l1"]["ls_auroc"]["diff_observed"],
            "p_value_holm_vs_l1": records[selected_cand]["diff_vs_l1"]["ls_auroc"]["p_value_holm"],
            "dev_auroc": records[selected_cand]["dev_auroc"],
            "pair_correct": records[selected_cand]["pair_correct"],
            "working_model_fallback": "R2",
        }
    else:
        print("RESULT: None of the candidates beat L1 on dev LS-AUROC with statistical significance (Holm p < 0.05).", flush=True)
        print("Model A does not exceed the length baseline with statistical significance.", flush=True)
        print("Mechanical rule action: SELECT NOTHING. Retaining R2 as the working model.", flush=True)

        selected_meta = {
            "selected": False,
            "candidate": None,
            "verdict": "None beats L1 on dev LS-AUROC with Holm p < 0.05",
            "working_model_retained": "R2",
            "working_model_checkpoint": "models/checkpoints/p4/r2_chunked_mean_cxx28.pkl",
        }

    # Save selection metadata
    with open(SELECTED_DIR / "selection_metadata.json", "w") as f:
        json.dump(selected_meta, f, indent=2)

    # Save complete Master Comparison artifact
    master_out = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
            "n_resamples": 2000,
            "step0_draw_std_ls_auroc": draw_std_ls,
        },
        "selection": selected_meta,
        "master_table": records,
    }

    out_file = RESULTS_P4_DIR / "master_comparison_p4.json"
    with open(out_file, "w") as f:
        json.dump(master_out, f, indent=2)

    print(f"\nSaved Master Comparison artifact to {out_file}", flush=True)
    print(f"Step 4 finished in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

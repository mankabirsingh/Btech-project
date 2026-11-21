"""STEP 2: Label-Noise Sensitivity and Conflict Sample Extraction.

Analyzes intra-train duplicate label conflicts under aggressive and light normalizations.
Extracts 30 random raw conflict pairs into results/p3/conflict_samples.md.
Evaluates N0 (uncleaned), N1 (light-cleaned), and N2 (aggressive-cleaned) data variants.
"""

from collections import defaultdict
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.p3.eval_protocol import (
    evaluate_candidate,
    get_cached_features,
    get_champion_p2_reference,
    load_datasets,
)

RESULTS_P3_DIR = PROJECT_ROOT / "results" / "p3"
RESULTS_P3_DIR.mkdir(parents=True, exist_ok=True)


def norm_light(code: str) -> str:
    """Light normalization: strip comments and collapse whitespace, preserving tokens."""
    c = re.sub(r"/\*.*?\*/", " ", str(code), flags=re.DOTALL)
    c = re.sub(r"//.*", " ", c)
    return " ".join(c.split())


def norm_aggressive(code: str) -> str:
    """Aggressive normalization: comments stripped, whitespace collapsed, STR, NUM, ID mapped."""
    c = re.sub(r"/\*.*?\*/", " ", str(code), flags=re.DOTALL)
    c = re.sub(r"//.*", " ", c)
    c = re.sub(r"\"(\\.|[^\"\\])*\"", " STR ", c)
    c = re.sub(r"\x27(\\.|[^\x27\\])*\x27", " STR ", c)
    c = re.sub(r"\b0x[0-9a-fA-F]+\b", " NUM ", c)
    c = re.sub(r"\b\d+(\.\d+)?([eE][+-]?\d+)?\b", " NUM ", c)

    cxx_kw = {
        "if", "else", "for", "while", "do", "switch", "case", "default", "return",
        "break", "continue", "goto", "sizeof", "typedef", "struct", "class", "union",
        "enum", "public", "private", "protected", "virtual", "override", "final",
        "const", "static", "volatile", "extern", "inline", "auto", "template",
        "typename", "namespace", "using", "try", "catch", "throw", "new", "delete",
        "nullptr", "NULL", "true", "false", "this", "operator", "int", "char",
        "float", "double", "void", "bool", "long", "short", "signed", "unsigned",
        "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t",
        "int16_t", "int32_t", "int64_t", "std"
    }

    toks = re.findall(r"\w+|[^\w\s]", c)
    norm_toks = []
    for t in toks:
        if re.match(r"^[A-Za-z_]\w*$", t):
            if t in cxx_kw or t in {"STR", "NUM"}:
                norm_toks.append(t)
            else:
                norm_toks.append("ID")
        else:
            norm_toks.append(t)
    return " ".join(norm_toks)


def main():
    print("=" * 80)
    print("STEP 2: LABEL-NOISE SENSITIVITY & CONFLICT PAIR EXTRACTION")
    print("=" * 80)
    t_start = time.time()

    df_train, df_dev, tok_train, tok_dev = load_datasets()
    features = get_cached_features()
    champ_ref = get_champion_p2_reference(df_train, df_dev, tok_dev, features)

    n_total = len(df_train)
    print(f"Loaded {n_total:,} train samples.")

    # 1. Normalization & Grouping
    print("\n[1] Computing Normalization Groups...")
    groups_light = defaultdict(list)
    groups_aggr = defaultdict(list)

    for idx, row in df_train.iterrows():
        code = row["code"]
        vuln = int(row["vuln_label"])
        nl = norm_light(code)
        na = norm_aggressive(code)
        groups_light[nl].append(idx)
        groups_aggr[na].append(idx)

    # Identify conflicts (groups containing both vuln=1 and vuln=0)
    y_train = df_train["vuln_label"].values.astype(int)

    def find_conflicts(groups_dict: Dict[str, List[int]]) -> List[List[int]]:
        conflicts = []
        for g_norm, indices in groups_dict.items():
            if len(indices) > 1:
                labels = y_train[indices]
                if len(np.unique(labels)) > 1:
                    conflicts.append(indices)
        return conflicts

    conflicts_light = find_conflicts(groups_light)
    conflicts_aggr = find_conflicts(groups_aggr)

    indices_conflict_light = set(idx for g in conflicts_light for idx in g)
    indices_conflict_aggr = set(idx for g in conflicts_aggr for idx in g)

    print(f"  Light Normalization Conflicts:      {len(conflicts_light):,} groups, {len(indices_conflict_light):,} samples ({len(indices_conflict_light)/n_total*100:.2f}%)")
    print(f"  Aggressive Normalization Conflicts: {len(conflicts_aggr):,} groups, {len(indices_conflict_aggr):,} samples ({len(indices_conflict_aggr)/n_total*100:.2f}%)")

    # 2. Token Length Distribution for Conflicting Groups
    tok_aggr_conflicts = tok_train[list(indices_conflict_aggr)]
    tok_dist = {
        "n_samples": int(len(tok_aggr_conflicts)),
        "min": int(np.min(tok_aggr_conflicts)) if len(tok_aggr_conflicts) > 0 else 0,
        "p25": float(np.percentile(tok_aggr_conflicts, 25)) if len(tok_aggr_conflicts) > 0 else 0.0,
        "median": float(np.median(tok_aggr_conflicts)) if len(tok_aggr_conflicts) > 0 else 0.0,
        "mean": float(np.mean(tok_aggr_conflicts)) if len(tok_aggr_conflicts) > 0 else 0.0,
        "p75": float(np.percentile(tok_aggr_conflicts, 75)) if len(tok_aggr_conflicts) > 0 else 0.0,
        "max": int(np.max(tok_aggr_conflicts)) if len(tok_aggr_conflicts) > 0 else 0,
        "std": float(np.std(tok_aggr_conflicts)) if len(tok_aggr_conflicts) > 0 else 0.0,
    }
    print(f"\n[2] Token Length Distribution of Aggressive Conflict Samples:")
    print(f"  Min: {tok_dist['min']}, P25: {tok_dist['p25']:.1f}, Median: {tok_dist['median']:.1f}, Mean: {tok_dist['mean']:.1f}, P75: {tok_dist['p75']:.1f}, Max: {tok_dist['max']}, Std: {tok_dist['std']:.1f}")

    # 3. Draw 30 Random Conflict Pairs into results/p3/conflict_samples.md
    print("\n[3] Drawing 30 Random Conflict Pairs into conflict_samples.md...")
    rng = np.random.RandomState(42)
    shuffled_groups = [g for g in conflicts_aggr]
    rng.shuffle(shuffled_groups)

    markdown_pairs = ["# Intra-Train Conflicting Normalized-Duplicate Samples\n"]
    markdown_pairs.append("> Note: Raw code pairs shown side-by-side with metadata. Unjudged for manual review.\n\n")

    pair_count = 0
    for g_idx, g_indices in enumerate(shuffled_groups):
        if pair_count >= 30:
            break
        vuln_sub = [idx for idx in g_indices if y_train[idx] == 1]
        safe_sub = [idx for idx in g_indices if y_train[idx] == 0]
        if not vuln_sub or not safe_sub:
            continue

        pair_count += 1
        v_idx = rng.choice(vuln_sub)
        s_idx = rng.choice(safe_sub)

        v_row = df_train.iloc[v_idx]
        s_row = df_train.iloc[s_idx]

        markdown_pairs.append(f"## Pair {pair_count} (Group Size: {len(g_indices)})\n")
        markdown_pairs.append(f"### Vulnerable Sample (vuln_label = 1)\n")
        markdown_pairs.append(f"- **Sample ID:** `{v_row['sample_id']}`\n")
        markdown_pairs.append(f"- **Project:** `{v_row['project']}`\n")
        markdown_pairs.append(f"- **Commit ID:** `{v_row['commit_id']}`\n")
        markdown_pairs.append(f"- **CWE:** `{v_row.get('cwe', 'N/A')}`\n")
        markdown_pairs.append(f"- **Tokens:** {tok_train[v_idx]}\n\n")
        markdown_pairs.append("```c\n" + str(v_row["code"]).strip() + "\n```\n\n")

        markdown_pairs.append(f"### Benign Sample (vuln_label = 0)\n")
        markdown_pairs.append(f"- **Sample ID:** `{s_row['sample_id']}`\n")
        markdown_pairs.append(f"- **Project:** `{s_row['project']}`\n")
        markdown_pairs.append(f"- **Commit ID:** `{s_row['commit_id']}`\n")
        markdown_pairs.append(f"- **Tokens:** {tok_train[s_idx]}\n\n")
        markdown_pairs.append("```c\n" + str(s_row["code"]).strip() + "\n```\n\n")
        markdown_pairs.append("---\n\n")

    md_out_path = RESULTS_P3_DIR / "conflict_samples.md"
    with open(md_out_path, "w", encoding="utf-8") as f:
        f.writelines(markdown_pairs)
    print(f"  Saved 30 conflict pairs to {md_out_path}")

    # 4. Construct Data Variants N0, N1, N2
    # N0 = original train_minus_dev
    # N1 = drop samples in light normalization conflict groups, redraw to 25k balanced
    # N2 = drop samples in aggressive normalization conflict groups, redraw to 25k balanced
    print("\n[4] Constructing Data Variants N0, N1, N2...")

    X_train_raw = np.hstack([features["train"]["emb_codebert_256"], features["train"]["cxx28"]])
    X_dev_raw = np.hstack([features["dev"]["emb_codebert_256"], features["dev"]["cxx28"]])

    def build_cleaned_variant(drop_indices: set, seed: int = 42) -> Tuple[np.ndarray, pd.DataFrame]:
        keep_mask = np.array([i not in drop_indices for i in range(n_total)])
        clean_df = df_train[keep_mask].copy().reset_index()  # preserves original index in 'index'
        orig_indices = clean_df["index"].values

        clean_vuln_pos = np.where(clean_df["vuln_label"] == 1)[0]
        clean_safe_pos = np.where(clean_df["vuln_label"] == 0)[0]

        # Resample with replacement to exactly 12,500 vuln + 12,500 safe
        rng_resample = np.random.RandomState(seed)
        chosen_vuln_pos = rng_resample.choice(clean_vuln_pos, size=12500, replace=True)
        chosen_safe_pos = rng_resample.choice(clean_safe_pos, size=12500, replace=True)

        chosen_pos = np.concatenate([chosen_vuln_pos, chosen_safe_pos])
        rng_resample.shuffle(chosen_pos)

        chosen_orig_indices = orig_indices[chosen_pos]
        df_variant = clean_df.iloc[chosen_pos].copy().reset_index(drop=True)

        # Re-partition 5-fold GroupKFold by project
        gkf = GroupKFold(n_splits=5)
        df_variant["kfold"] = -1
        for fold, (tr_i, val_i) in enumerate(gkf.split(df_variant, df_variant["vuln_label"], groups=df_variant["project"])):
            df_variant.loc[val_i, "kfold"] = fold

        X_variant = X_train_raw[chosen_orig_indices]
        return X_variant, df_variant

    X_N0 = X_train_raw
    df_N0 = df_train

    X_N1, df_N1 = build_cleaned_variant(indices_conflict_light, seed=42)
    X_N2, df_N2 = build_cleaned_variant(indices_conflict_aggr, seed=42)

    data_variants = [
        ("N0", "No cleaning (champion_p2 data)", X_N0, df_N0),
        ("N1", "Drop light-normalization conflict samples, redrawn to 25k balanced", X_N1, df_N1),
        ("N2", "Drop aggressive-normalization conflict samples, redrawn to 25k balanced", X_N2, df_N2),
    ]

    # 5. Retrain Champion Recipe on N0, N1, N2
    results_variants = {}
    for vid, desc, x_tr, df_tr in data_variants:
        print(f"\n--- Retraining Champion Recipe on {vid}: {desc} ---")
        res = evaluate_candidate(
            candidate_id=vid,
            description=desc,
            category="Data Variant",
            X_train_raw=x_tr,
            X_dev_raw=X_dev_raw,
            df_train=df_tr,
            df_dev=df_dev,
            tok_dev=tok_dev,
            champ_ref=champ_ref,
            C_val=0.01,
        )
        print(f"  Dev AUROC: {res['dev_auroc']:.4f} {res['dev_auroc_ci_95']}")
        print(f"  Dev Length-Stratified AUROC: {res['dev_length_stratified_auroc']:.4f} {res['dev_length_stratified_ci_95']}")
        print(f"  Dev C++ AUROC: {res['dev_cpp_auroc']:.4f} {res['dev_cpp_auroc_ci_95']}")
        print(f"  CV Macro AUROC: {res['cv_mean_auroc']:.4f} +/- {res['cv_std_auroc']:.4f}")
        results_variants[vid] = res

    # 6. Select Best Data Variant by dev length-stratified AUROC
    best_vid = "N0"
    best_ls_auc = results_variants["N0"]["dev_length_stratified_auroc"]

    for vid in ["N1", "N2"]:
        ls_auc = results_variants[vid]["dev_length_stratified_auroc"]
        # Check if significantly different from N0
        diff_info = results_variants[vid]["diff_vs_champion"]["length_stratified_auroc"]
        p_val = diff_info["p_value"]
        if ls_auc > best_ls_auc and p_val < 0.05:
            best_vid = vid
            best_ls_auc = ls_auc

    print(f"\nBest Data Variant Selected: {best_vid} (Dev Length-Stratified AUROC = {best_ls_auc:.4f})")

    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
        },
        "conflict_statistics": {
            "light_normalization": {
                "groups": len(conflicts_light),
                "samples": len(indices_conflict_light),
                "pct_of_train": float(len(indices_conflict_light) / n_total * 100),
            },
            "aggressive_normalization": {
                "groups": len(conflicts_aggr),
                "samples": len(indices_conflict_aggr),
                "pct_of_train": float(len(indices_conflict_aggr) / n_total * 100),
                "token_distribution": tok_dist,
            },
            "conflict_samples_file": "results/p3/conflict_samples.md",
        },
        "variants": {
            k: {sk: sv for sk, sv in v.items() if sk not in ["p_dev", "p_oof"]}
            for k, v in results_variants.items()
        },
        "best_variant": best_vid,
    }

    out_json = RESULTS_P3_DIR / "step2_label_noise.json"
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nStep 2 successfully saved to {out_json} in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

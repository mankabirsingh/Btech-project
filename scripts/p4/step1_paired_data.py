"""Step 1: Paired Data Check and dev_pairs.parquet Construction."""

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
from sklearn.preprocessing import StandardScaler

from scripts.p3.eval_protocol import get_cached_features, load_datasets
from src.eval.stats import bootstrap_ci

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
P2_SPLITS_DIR = SPLITS_DIR / "p2"
P4_SPLITS_DIR = SPLITS_DIR / "p4"
RESULTS_P4_DIR = PROJECT_ROOT / "results" / "p4"

P4_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_P4_DIR.mkdir(parents=True, exist_ok=True)


def get_func_name(code: str) -> str:
    """Extract C/C++ function name from definition header."""
    m = re.search(r"\b([a-zA-Z_~][a-zA-Z0-9_]*)\s*\([^;{}]*\)\s*(?:const)?\s*(?:noexcept)?\s*\{", str(code))
    return m.group(1) if m else "unknown"


def extract_disjoint_pairs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Extracts 1-to-1 disjoint pairs of (vulnerable, fixed) functions.
    Priority 1: parent_sample_id link.
    Priority 2: 1-to-1 matching by (commit_id, func_name) where code differs.
    """
    id_to_row = {r["sample_id"]: r for _, r in df.iterrows()}
    used_vuln = set()
    used_safe = set()
    pairs = []

    # Priority 1: parent_sample_id
    for _, r in df.iterrows():
        p_id = r["parent_sample_id"]
        if pd.notna(p_id) and p_id in id_to_row:
            parent = id_to_row[p_id]
            if r["vuln_label"] != parent["vuln_label"]:
                v = r if r["vuln_label"] == 1 else parent
                s = parent if r["vuln_label"] == 1 else r
                v_id, s_id = v["sample_id"], s["sample_id"]
                if v_id not in used_vuln and s_id not in used_safe and v["code"] != s["code"]:
                    used_vuln.add(v_id)
                    used_safe.add(s_id)
                    pairs.append({
                        "vuln_sample_id": v_id,
                        "fixed_sample_id": s_id,
                        "vuln_code": v["code"],
                        "fixed_code": s["code"],
                        "project": v["project"],
                        "commit_id": v["commit_id"],
                        "func_name": get_func_name(v["code"]),
                        "source_method": "parent_sample_id",
                    })

    # Priority 2: 1-to-1 (commit_id, func_name)
    for cid, g in df.groupby("commit_id"):
        vulns = g[g["vuln_label"] == 1]
        safes = g[g["vuln_label"] == 0]
        if len(vulns) > 0 and len(safes) > 0:
            v_by_name = {}
            for _, v in vulns.iterrows():
                if v["sample_id"] in used_vuln:
                    continue
                vn = get_func_name(v["code"])
                if vn and vn != "unknown":
                    v_by_name.setdefault(vn, []).append(v)
            s_by_name = {}
            for _, s in safes.iterrows():
                if s["sample_id"] in used_safe:
                    continue
                sn = get_func_name(s["code"])
                if sn and sn != "unknown":
                    s_by_name.setdefault(sn, []).append(s)

            for name, v_list in v_by_name.items():
                if name in s_by_name:
                    s_list = s_by_name[name]
                    if len(v_list) == 1 and len(s_list) == 1:
                        v = v_list[0]
                        s = s_list[0]
                        v_id, s_id = v["sample_id"], s["sample_id"]
                        if v_id not in used_vuln and s_id not in used_safe and v["code"] != s["code"]:
                            used_vuln.add(v_id)
                            used_safe.add(s_id)
                            pairs.append({
                                "vuln_sample_id": v_id,
                                "fixed_sample_id": s_id,
                                "vuln_code": v["code"],
                                "fixed_code": s["code"],
                                "project": v["project"],
                                "commit_id": cid,
                                "func_name": name,
                                "source_method": "commit_func_name",
                            })

    return pairs


def compute_paired_metrics(
    p_vuln: np.ndarray,
    p_fixed: np.ndarray,
    threshold: float = 0.5,
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes pair_correct, P-C, P-V, P-B, P-R with 2000-resample bootstrap CIs."""
    n = len(p_vuln)
    is_correct = (p_vuln > p_fixed).astype(float)
    is_pc = ((p_vuln >= threshold) & (p_fixed < threshold)).astype(float)
    is_pv = ((p_vuln >= threshold) & (p_fixed >= threshold)).astype(float)
    is_pb = ((p_vuln < threshold) & (p_fixed < threshold)).astype(float)
    is_pr = ((p_vuln < threshold) & (p_fixed >= threshold)).astype(float)

    rng = np.random.RandomState(seed)
    boot_correct, boot_pc, boot_pv, boot_pb, boot_pr = [], [], [], [], []

    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot_correct.append(np.mean(is_correct[idx]))
        boot_pc.append(np.mean(is_pc[idx]))
        boot_pv.append(np.mean(is_pv[idx]))
        boot_pb.append(np.mean(is_pb[idx]))
        boot_pr.append(np.mean(is_pr[idx]))

    lower = 2.5
    upper = 97.5

    return {
        "pair_correct": float(np.mean(is_correct)),
        "pair_correct_ci_95": [float(np.percentile(boot_correct, lower)), float(np.percentile(boot_correct, upper))],
        "P-C": float(np.mean(is_pc)),
        "P-C_ci_95": [float(np.percentile(boot_pc, lower)), float(np.percentile(boot_pc, upper))],
        "P-V": float(np.mean(is_pv)),
        "P-V_ci_95": [float(np.percentile(boot_pv, lower)), float(np.percentile(boot_pv, upper))],
        "P-B": float(np.mean(is_pb)),
        "P-B_ci_95": [float(np.percentile(boot_pb, lower)), float(np.percentile(boot_pb, upper))],
        "P-R": float(np.mean(is_pr)),
        "P-R_ci_95": [float(np.percentile(boot_pr, lower)), float(np.percentile(boot_pr, upper))],
        "threshold": float(threshold),
        "n_pairs": n,
    }


def main():
    t_start = time.time()
    print("=" * 80)
    print("STEP 1: PAIRED DATA CHECK AND EVALUATION")
    print("=" * 80)

    # 1. Inspect datasets and check column availability
    df_train, df_dev, tok_train, tok_dev = load_datasets()
    print(f"\n1.1 Column Audit:")
    print(f"  Dev columns: {list(df_dev.columns)}")
    has_func_after = "func_after" in df_dev.columns or "func_before" in df_dev.columns
    has_fixed_code = "fixed_code" in df_dev.columns
    print(f"  Direct post-fix columns ('func_after' / 'func_before'): {has_func_after}")
    print(f"  Fixed code field ('fixed_code'): {has_fixed_code}")
    print(f"  Pairing via parent_sample_id available: {'parent_sample_id' in df_dev.columns}")
    print(f"  Pairing via (commit_id, func_name) available: True (commit_id present, functions extracted)")

    # 1.2 Count pairs in dev_heldout
    print("\n  Extracting pairs from dev_heldout...")
    dev_pairs = extract_disjoint_pairs(df_dev)
    n_dev_pairs = len(dev_pairs)
    print(f"  Total 1-to-1 disjoint pairs found in dev_heldout: {n_dev_pairs}")

    # Breakdown by method
    dev_parent_count = sum(1 for p in dev_pairs if p["source_method"] == "parent_sample_id")
    dev_commit_count = sum(1 for p in dev_pairs if p["source_method"] == "commit_func_name")
    print(f"    - parent_sample_id links: {dev_parent_count}")
    print(f"    - commit_id + func_name: {dev_commit_count}")

    # Check length statistics within pairs
    dev_id_to_tok = dict(zip(df_dev["sample_id"], tok_dev))
    for p in dev_pairs:
        p["vuln_tokens"] = int(dev_id_to_tok.get(p["vuln_sample_id"], len(str(p["vuln_code"]).split())))
        p["fixed_tokens"] = int(dev_id_to_tok.get(p["fixed_sample_id"], len(str(p["fixed_code"]).split())))

    df_dev_pairs = pd.DataFrame(dev_pairs)
    df_dev_pairs["pair_id"] = [f"dev_pair_{i:04d}" for i in range(len(df_dev_pairs))]
    pairs_file = P4_SPLITS_DIR / "dev_pairs.parquet"
    df_dev_pairs.to_parquet(pairs_file, index=False)
    print(f"  Saved dev pairs to {pairs_file} (n={len(df_dev_pairs)})")

    # 1.3 Count pairs in train_pool (clean source pool for train_minus_dev)
    print("\n  Auditing training pool for pairs...")
    df_expanded = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    TEST_PROJECTS = {"envoy", "openssl", "vim", "gpac"}
    EXCLUDED_PROJECTS = {"chrome", "chromium", "tensorflow"}
    DEV_PROJECTS = {"qpdf", "openexr", "tcpdump", "radare2"}
    all_excluded = TEST_PROJECTS | EXCLUDED_PROJECTS | DEV_PROJECTS

    pool_mask = ~df_expanded["project"].str.lower().isin(all_excluded)
    df_pool = df_expanded[pool_mask].copy().reset_index(drop=True)
    print(f"  Full clean training source pool size: {len(df_pool)} samples")

    pool_pairs = extract_disjoint_pairs(df_pool)
    n_pool_pairs = len(pool_pairs)
    pool_parent_count = sum(1 for p in pool_pairs if p["source_method"] == "parent_sample_id")
    pool_commit_count = sum(1 for p in pool_pairs if p["source_method"] == "commit_func_name")
    print(f"  Total 1-to-1 disjoint pairs found in train_pool: {n_pool_pairs}")
    print(f"    - parent_sample_id links: {pool_parent_count}")
    print(f"    - commit_id + func_name: {pool_commit_count}")

    # Pairs in train_minus_dev 25k subset
    train_sub_pairs = extract_disjoint_pairs(df_train)
    n_train_sub_pairs = len(train_sub_pairs)
    print(f"  Total 1-to-1 disjoint pairs found in train_minus_dev (25k subset): {n_train_sub_pairs}")

    # Length analysis of dev pairs
    v_toks = df_dev_pairs["vuln_tokens"].values
    f_toks = df_dev_pairs["fixed_tokens"].values
    frac_fixed_longer = float(np.mean(f_toks > v_toks))
    frac_vuln_longer = float(np.mean(v_toks > f_toks))
    frac_equal = float(np.mean(v_toks == f_toks))
    print(f"\n  Length Directionality in dev_pairs:")
    print(f"    Fraction where fixed code is strictly longer: {frac_fixed_longer * 100:.2f}%")
    print(f"    Fraction where vulnerable code is strictly longer: {frac_vuln_longer * 100:.2f}%")
    print(f"    Fraction where lengths are identical: {frac_equal * 100:.2f}%")

    # 1.4 Evaluate Baselines on dev_pairs
    print("\n1.2 Paired Metrics on dev_pairs:")
    feats = get_cached_features()

    # 1.4.1 L1 Baseline (log n_tokens only)
    log_tok_tr = np.log1p(tok_train).reshape(-1, 1).astype(np.float32)
    log_tok_dv = np.log1p(tok_dev).reshape(-1, 1).astype(np.float32)
    sc_l1 = StandardScaler()
    clf_l1 = LogisticRegression(C=0.01, class_weight="balanced", random_state=42)
    clf_l1.fit(sc_l1.fit_transform(log_tok_tr), df_train["vuln_label"])
    p_dv_l1 = clf_l1.predict_proba(sc_l1.transform(log_tok_dv))[:, 1]

    # 1.4.2 Faithful Champion P2 (CodeBERT + scaled cxx28, C=0.003)
    sc_cxx = StandardScaler()
    sc_cxx_tr = sc_cxx.fit_transform(feats["train"]["cxx28"])
    sc_cxx_dv = sc_cxx.transform(feats["dev"]["cxx28"])
    X_tr_champ = np.hstack([feats["train"]["emb_codebert_256"], sc_cxx_tr])
    X_dv_champ = np.hstack([feats["dev"]["emb_codebert_256"], sc_cxx_dv])
    clf_champ = LogisticRegression(C=0.003, class_weight="balanced", max_iter=1000, random_state=42)
    clf_champ.fit(X_tr_champ, df_train["vuln_label"])
    p_dv_champ = clf_champ.predict_proba(X_dv_champ)[:, 1]

    # 1.4.3 R2 (Chunked-mean CodeBERT + scaled cxx28, C=0.003)
    X_tr_r2 = np.hstack([feats["train"]["emb_chunked_mean"], sc_cxx_tr])
    X_dv_r2 = np.hstack([feats["dev"]["emb_chunked_mean"], sc_cxx_dv])
    clf_r2 = LogisticRegression(C=0.003, class_weight="balanced", max_iter=1000, random_state=42)
    clf_r2.fit(X_tr_r2, df_train["vuln_label"])
    p_dv_r2 = clf_r2.predict_proba(X_dv_r2)[:, 1]

    dev_id_to_idx = {sid: i for i, sid in enumerate(df_dev["sample_id"])}
    v_idx = [dev_id_to_idx[sid] for sid in df_dev_pairs["vuln_sample_id"]]
    f_idx = [dev_id_to_idx[sid] for sid in df_dev_pairs["fixed_sample_id"]]

    metrics_l1 = compute_paired_metrics(p_dv_l1[v_idx], p_dv_l1[f_idx], threshold=0.5)
    metrics_champ = compute_paired_metrics(p_dv_champ[v_idx], p_dv_champ[f_idx], threshold=0.5)
    metrics_r2 = compute_paired_metrics(p_dv_r2[v_idx], p_dv_r2[f_idx], threshold=0.5)

    print(f"  L1 Baseline:")
    print(f"    pair_correct: {metrics_l1['pair_correct']*100:.2f}% (95% CI: [{metrics_l1['pair_correct_ci_95'][0]*100:.2f}%, {metrics_l1['pair_correct_ci_95'][1]*100:.2f}%])")
    print(f"    P-C: {metrics_l1['P-C']*100:.2f}%, P-V: {metrics_l1['P-V']*100:.2f}%, P-B: {metrics_l1['P-B']*100:.2f}%, P-R: {metrics_l1['P-R']*100:.2f}%")

    print(f"  Faithful Champion P2:")
    print(f"    pair_correct: {metrics_champ['pair_correct']*100:.2f}% (95% CI: [{metrics_champ['pair_correct_ci_95'][0]*100:.2f}%, {metrics_champ['pair_correct_ci_95'][1]*100:.2f}%])")
    print(f"    P-C: {metrics_champ['P-C']*100:.2f}%, P-V: {metrics_champ['P-V']*100:.2f}%, P-B: {metrics_champ['P-B']*100:.2f}%, P-R: {metrics_champ['P-R']*100:.2f}%")

    print(f"  R2 Candidate (Chunked-Mean):")
    print(f"    pair_correct: {metrics_r2['pair_correct']*100:.2f}% (95% CI: [{metrics_r2['pair_correct_ci_95'][0]*100:.2f}%, {metrics_r2['pair_correct_ci_95'][1]*100:.2f}%])")
    print(f"    P-C: {metrics_r2['P-C']*100:.2f}%, P-V: {metrics_r2['P-V']*100:.2f}%, P-B: {metrics_r2['P-B']*100:.2f}%, P-R: {metrics_r2['P-R']*100:.2f}%")

    # Save artifact
    step1_out = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
            "dev_pairs_path": str(pairs_file),
        },
        "counts": {
            "dev_heldout_total_samples": len(df_dev),
            "dev_pairs_count": n_dev_pairs,
            "dev_pairs_parent_sample_id": dev_parent_count,
            "dev_pairs_commit_func_name": dev_commit_count,
            "train_pool_total_samples": len(df_pool),
            "train_pool_pairs_count": n_pool_pairs,
            "train_pool_parent_sample_id": pool_parent_count,
            "train_pool_commit_func_name": pool_commit_count,
            "train_minus_dev_subset_pairs": n_train_sub_pairs,
        },
        "length_directionality": {
            "frac_fixed_longer": frac_fixed_longer,
            "frac_vuln_longer": frac_vuln_longer,
            "frac_equal": frac_equal,
            "mean_tokens_vuln": float(np.mean(v_toks)),
            "mean_tokens_fixed": float(np.mean(f_toks)),
        },
        "baselines_on_dev_pairs": {
            "L1": metrics_l1,
            "faithful_champion_p2": metrics_champ,
            "R2": metrics_r2,
        },
    }

    out_file = RESULTS_P4_DIR / "step1_paired_data.json"
    with open(out_file, "w") as f:
        json.dump(step1_out, f, indent=2)

    print(f"\nSaved Step 1 results to {out_file}")
    print(f"Step 1 completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

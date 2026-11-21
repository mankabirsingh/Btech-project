#!/usr/bin/env python3
"""
Build training datasets for Real-World Model A v2 (Part A1.2):
1. train_v2.parquet:
   - 25,000 balanced sample drawn using Phase 15 recipe (12.5k vuln, 12.5k safe, seed 42)
   - Excludes:
     * Dev projects (qpdf, openexr, tcpdump, radare2)
     * Old test projects (envoy, openssl, vim, gpac)
     * New test2 projects (node, Bento4, MilkyTracker, cryptopp, libvpx, iperf, etc.)
     * Circular/oversized projects (chromium, tensorflow, chrome)
     * Near-duplicates to dev, old test, and test2 (MinHash Jaccard >= 0.8)
2. train_pairs_v2.parquet:
   - Disjoint vulnerable/fixed function pairs from remaining clean pool
   - Excludes any pair with near-duplicates to dev, old test, or test2

Saves:
- data/p10/train_v2.parquet
- data/p10/train_pairs_v2.parquet
- results/p10/datasets_v2_summary.json
"""

import sys
import os
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.p4.step1_paired_data import extract_disjoint_pairs

SOURCE_TRAIN_EXPANDED = PROJECT_ROOT / "data" / "splits" / "vulnerability_train_expanded.parquet"
DATA_P10_DIR = PROJECT_ROOT / "data" / "p10"
RESULTS_DIR = PROJECT_ROOT / "results" / "p10"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

TRAIN_V2_PARQUET = DATA_P10_DIR / "train_v2.parquet"
TRAIN_PAIRS_V2_PARQUET = DATA_P10_DIR / "train_pairs_v2.parquet"
SUMMARY_JSON = RESULTS_DIR / "datasets_v2_summary.json"

DEV_PROJECTS = {"qpdf", "openexr", "tcpdump", "radare2"}
OLD_TEST_PROJECTS = {"envoy", "openssl", "vim", "gpac"}
TOO_BIG_PROJECTS = {"chromium", "tensorflow", "chrome"}


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    t0 = time.time()
    print("=" * 80)
    print("BUILDING REAL-WORLD MODEL A V2 DATASETS (train_v2 and train_pairs_v2)")
    print("=" * 80)
    DATA_P10_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load test2 metadata
    test2_meta_path = RESULTS_DIR / "test2_projects.json"
    assert test2_meta_path.exists(), f"Missing {test2_meta_path}! Run build_test2_dataset.py first."
    with open(test2_meta_path) as f:
        test2_meta = json.load(f)
    test2_projects = set(test2_meta["selected_summary"]["cpp_projects"] + test2_meta["selected_summary"]["c_projects"])
    print(f"[1] Test2 projects to exclude: {sorted(list(test2_projects))}")

    all_excluded_projects = {p.lower() for p in (DEV_PROJECTS | OLD_TEST_PROJECTS | TOO_BIG_PROJECTS | test2_projects)}

    # 2. Load expanded train pool
    print("[2] Loading vulnerability_train_expanded.parquet...")
    table = pq.read_table(
        SOURCE_TRAIN_EXPANDED,
        columns=["sample_id", "project", "commit_id", "language", "vuln_label", "code", "cwe", "parent_sample_id"]
    )
    df_pool = table.to_pandas()
    print(f"    Total rows in training pool: {len(df_pool)}")

    # 3. Filter by project
    pool_mask = ~df_pool["project"].str.lower().isin(all_excluded_projects)
    df_clean_projs = df_pool[pool_mask].copy().reset_index(drop=True)
    print(f"    Rows after project exclusion: {len(df_clean_projs)}")

    # 4. Load near-duplicates
    print("[3] Loading near-duplicate mappings (Jaccard >= 0.8)...")
    dup_test_file = PROJECT_ROOT / "results" / "p2" / "near_dups_test_train.csv"
    dup_val_file = PROJECT_ROOT / "results" / "p2" / "near_dups_val_train.csv"

    dup_test = pd.read_csv(dup_test_file)
    dup_val = pd.read_csv(dup_val_file)

    old_test_dup_ids = set(dup_test[dup_test["jaccard"] >= 0.8]["train_sample_id"])
    dev_dup_ids = set(dup_val[dup_val["jaccard"] >= 0.8]["train_sample_id"])
    print(f"    Near-duplicates to old test: {len(old_test_dup_ids)}")
    print(f"    Near-duplicates to dev:      {len(dev_dup_ids)}")

    # Find near-duplicates to test2 using cached train_minhash_128
    print("    Finding near-duplicates to test2...")
    train_minhash = np.load(CACHE_DIR / "train_minhash_128.npy")
    test2_indices = df_pool[df_pool["project"].isin(test2_projects)].index.values

    lsh_test2 = MinHashLSH(threshold=0.8, num_perm=128)
    for idx in test2_indices:
        mh = MinHash(num_perm=128, hashvalues=train_minhash[idx], scheme="affine32")
        lsh_test2.insert(str(idx), mh)

    clean_proj_orig_indices = df_pool[pool_mask].index.values
    test2_dup_ids = set()
    for orig_idx in clean_proj_orig_indices:
        mh = MinHash(num_perm=128, hashvalues=train_minhash[orig_idx], scheme="affine32")
        matches = lsh_test2.query(mh)
        if matches:
            test2_dup_ids.add(df_pool.iloc[orig_idx]["sample_id"])
    print(f"    Near-duplicates to test2:    {len(test2_dup_ids)}")

    all_dup_sample_ids = old_test_dup_ids | dev_dup_ids | test2_dup_ids
    print(f"    Total unique near-duplicate sample IDs: {len(all_dup_sample_ids)}")

    # 5. Filter out near-duplicates from pool
    df_clean_pool = df_clean_projs[~df_clean_projs["sample_id"].isin(all_dup_sample_ids)].copy().reset_index(drop=True)
    n_clean_pool = len(df_clean_pool)
    n_vuln_pool = int((df_clean_pool["vuln_label"] == 1).sum())
    n_safe_pool = int((df_clean_pool["vuln_label"] == 0).sum())
    print(f"[4] Clean training pool: {n_clean_pool} samples ({n_vuln_pool} vuln, {n_safe_pool} safe)")

    # 6. Build train_v2 (Phase 15 balanced recipe: 12.5k vuln, 12.5k safe, seed 42)
    print("[5] Sampling 25,000 balanced train_v2 (seed 42)...")
    vuln_clean = df_clean_pool[df_clean_pool["vuln_label"] == 1]
    safe_clean = df_clean_pool[df_clean_pool["vuln_label"] == 0]

    assert len(vuln_clean) >= 12500, f"Insufficient vulnerable samples: {len(vuln_clean)} < 12500"
    assert len(safe_clean) >= 12500, f"Insufficient safe samples: {len(safe_clean)} < 12500"

    v_sampled = vuln_clean.sample(n=12500, random_state=42)
    s_sampled = safe_clean.sample(n=12500, random_state=42)

    df_train_v2 = pd.concat([v_sampled, s_sampled], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"    train_v2 created: {len(df_train_v2)} rows (Prevalence: {df_train_v2['vuln_label'].mean():.4f})")
    df_train_v2.to_parquet(TRAIN_V2_PARQUET, index=False)
    sha_train_v2 = compute_file_sha256(TRAIN_V2_PARQUET)
    print(f"    Saved train_v2 to {TRAIN_V2_PARQUET} (SHA-256: {sha_train_v2})")

    # 7. Build train_pairs_v2
    print("[6] Extracting disjoint vulnerable/fixed pairs from clean pool...")
    raw_pairs = extract_disjoint_pairs(df_clean_pool)
    print(f"    Extracted {len(raw_pairs)} raw disjoint pairs.")

    clean_pairs = []
    for p in raw_pairs:
        v_id = p["vuln_sample_id"]
        s_id = p["fixed_sample_id"]
        if v_id in all_dup_sample_ids or s_id in all_dup_sample_ids:
            continue
        clean_pairs.append(p)

    df_pairs_v2 = pd.DataFrame(clean_pairs)
    print(f"    train_pairs_v2 created: {len(df_pairs_v2)} clean pairs.")
    df_pairs_v2.to_parquet(TRAIN_PAIRS_V2_PARQUET, index=False)
    sha_pairs_v2 = compute_file_sha256(TRAIN_PAIRS_V2_PARQUET)
    print(f"    Saved train_pairs_v2 to {TRAIN_PAIRS_V2_PARQUET} (SHA-256: {sha_pairs_v2})")

    # 8. Save summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "clean_pool_size": n_clean_pool,
        "clean_pool_vuln": n_vuln_pool,
        "clean_pool_safe": n_safe_pool,
        "train_v2": {
            "path": str(TRAIN_V2_PARQUET),
            "n_samples": len(df_train_v2),
            "prevalence": float(df_train_v2["vuln_label"].mean()),
            "sha256": sha_train_v2,
        },
        "train_pairs_v2": {
            "path": str(TRAIN_PAIRS_V2_PARQUET),
            "n_pairs": len(df_pairs_v2),
            "sha256": sha_pairs_v2,
        },
        "exclusions": {
            "dev_projects": list(DEV_PROJECTS),
            "old_test_projects": list(OLD_TEST_PROJECTS),
            "test2_projects": list(test2_projects),
            "near_dups_dev_count": len(dev_dup_ids),
            "near_dups_old_test_count": len(old_test_dup_ids),
            "near_dups_test2_count": len(test2_dup_ids),
            "total_near_dups_count": len(all_dup_sample_ids),
        }
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[7] Dataset summary written to {SUMMARY_JSON} in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()

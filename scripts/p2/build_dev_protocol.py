"""Project-held-out development split construction and cross-validation protocol."""

import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import GroupKFold

from scripts.p2.code_normalizer import CPP_KEYWORDS
from src.eval.locked_test import compute_file_sha256

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
P2_SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "p2"
RESULTS_P2_DIR = PROJECT_ROOT / "results" / "p2"

TEST_PROJECTS = {"envoy", "openssl", "vim", "gpac"}
EXCLUDED_PROJECTS = {"chrome", "chromium", "tensorflow"}
DEV_PROJECTS = ["qpdf", "openexr", "tcpdump", "radare2"]


def check_cpp_syntax(code: str) -> bool:
    """Fast regex-free heuristic checking for modern C++ tokens and keywords."""
    if not isinstance(code, str) or not code:
        return False
    # Check for direct C++ specific operators and tokens
    if "std::" in code or "::" in code or "nullptr" in code or "template<" in code or "template <" in code:
        return True
    # Fast token membership
    tokens = code.split()
    cpp_specific = {
        "class", "namespace", "public:", "private:", "protected:", "constexpr",
        "unique_ptr", "shared_ptr", "static_cast", "dynamic_cast", "reinterpret_cast",
        "typename", "explicit", "override", "noexcept", "decltype"
    }
    for t in tokens:
        if t in cpp_specific:
            return True
    return False


def main():
    t_start = time.time()
    P2_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_P2_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 5: PROJECT-HELD-OUT DEVELOPMENT PROTOCOL")
    print("=" * 80)

    # 1. Load Training Pool
    print("\n[1] Loading Full Expanded Training Pool (138,412 rows)...")
    train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    print(f"  Loaded {len(train_df):,} samples.")

    # 2. Project Catalog Analysis
    print("\n[2] Computing project summary: sample count, % vuln, % C++...")
    train_df["is_cpp"] = [check_cpp_syntax(c) for c in train_df["code"].values]

    proj_stats = train_df.groupby("project").agg(
        total_samples=("sample_id", "count"),
        vuln_samples=("vuln_label", "sum"),
        cpp_samples=("is_cpp", "sum")
    ).reset_index()

    proj_stats["vuln_rate"] = proj_stats["vuln_samples"] / proj_stats["total_samples"]
    proj_stats["cpp_rate"] = proj_stats["cpp_samples"] / proj_stats["total_samples"]
    proj_stats["in_test"] = proj_stats["project"].str.lower().isin(TEST_PROJECTS)
    proj_stats["excluded"] = proj_stats["project"].str.lower().isin(EXCLUDED_PROJECTS)

    # Save complete project catalog
    catalog_path = RESULTS_P2_DIR / "training_pool_projects.csv"
    proj_stats.sort_values(by="total_samples", ascending=False).to_csv(catalog_path, index=False)
    print(f"  Saved project catalog to {catalog_path}")

    # 3. Form Held-Out Dev Set
    print("\n[3] Isolating Dev Set Projects: " + ", ".join(DEV_PROJECTS))
    dev_mask = train_df["project"].isin(DEV_PROJECTS)
    dev_df = train_df[dev_mask].copy().reset_index(drop=True)

    n_dev = len(dev_df)
    n_dev_vuln = int(dev_df["vuln_label"].sum())
    dev_prev = float(n_dev_vuln / n_dev)
    n_dev_cpp = int(dev_df["is_cpp"].sum())
    dev_cpp_rate = float(n_dev_cpp / n_dev)

    print(f"  Dev Set Size:       {n_dev:,} samples")
    print(f"  Dev Prevalence:     {dev_prev*100:.2f}% ({n_dev_vuln:,} vulnerable)")
    print(f"  Dev C++ Samples:    {n_dev_cpp:,} ({dev_cpp_rate*100:.2f}%)")

    # 4. Remove Near-Duplicates of Dev from Remaining Training Pool
    print("\n[4] Detecting & removing near-duplicates of Dev in remaining training pool...")
    # Candidate pool: strictly exclude Dev projects, Test projects, and Excluded projects
    non_dev_mask = (
        ~train_df["project"].isin(DEV_PROJECTS) &
        ~train_df["project"].str.lower().isin(TEST_PROJECTS) &
        ~train_df["project"].str.lower().isin(EXCLUDED_PROJECTS)
    )
    rem_indices = np.where(non_dev_mask)[0]
    dev_indices = np.where(dev_mask)[0]
    rem_pool_df = train_df.iloc[rem_indices].copy().reset_index(drop=True)
    print(f"  Eligible training pool before dedup: {len(rem_pool_df):,} samples")

    # Load precomputed train minhashes from cache
    train_hvals_path = PROJECT_ROOT / "data" / "cache" / "train_minhash_128.npy"
    if train_hvals_path.exists():
        print(f"  Loading precomputed train MinHashes from {train_hvals_path.name}...")
        train_hvals = np.load(train_hvals_path)
        dev_hvals = train_hvals[dev_indices]
        rem_hvals = train_hvals[rem_indices]
    else:
        raise FileNotFoundError(f"Missing {train_hvals_path}. Please run Step 4 first.")

    # Index Dev in LSH
    dev_lsh = MinHashLSH(threshold=0.80, num_perm=128)
    dev_mhs = []
    for d_idx in range(len(dev_hvals)):
        dmh = MinHash(num_perm=128, hashvalues=dev_hvals[d_idx], scheme="affine32")
        dev_mhs.append(dmh)
        dev_lsh.insert(d_idx, dmh)

    # Query remaining pool against Dev LSH
    print("  Querying remaining pool against Dev LSH...")
    t0 = time.time()
    leaked_rem_indices = set()
    for r_idx in range(len(rem_hvals)):
        rmh = MinHash(num_perm=128, hashvalues=rem_hvals[r_idx], scheme="affine32")
        candidates = dev_lsh.query(rmh)
        for c_idx in candidates:
            if rmh.jaccard(dev_mhs[c_idx]) >= 0.80:
                leaked_rem_indices.add(r_idx)
                break

    print(f"  Found {len(leaked_rem_indices)} samples in remaining pool near-duplicate to Dev set (in {time.time()-t0:.2f}s).")
    clean_pool_df = rem_pool_df[~rem_pool_df.index.isin(leaked_rem_indices)].copy().reset_index(drop=True)
    print(f"  Clean remaining pool: {len(clean_pool_df):,} samples")

    # 5. Re-draw 25,000 Balanced Training Split (Phase 15 Recipe)
    print("\n[5] Re-drawing 25,000 Balanced Training Split (12,500 vuln + 12,500 safe, seed=42)...")
    vuln_clean = clean_pool_df[clean_pool_df["vuln_label"] == 1]
    safe_clean = clean_pool_df[clean_pool_df["vuln_label"] == 0]

    print(f"  Available clean: {len(vuln_clean):,} vulnerable, {len(safe_clean):,} safe")
    train_25k = pd.concat([
        vuln_clean.sample(n=12500, random_state=42),
        safe_clean.sample(n=12500, random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 6. Apply 5-Fold GroupKFold by Project
    print("\n[6] Partitioning train_minus_dev with 5-Fold GroupKFold by project...")
    gkf = GroupKFold(n_splits=5)
    train_25k["kfold"] = -1
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_25k, train_25k["vuln_label"], groups=train_25k["project"])):
        train_25k.loc[val_idx, "kfold"] = fold

    print("  GroupKFold distribution across folds:")
    for f in range(5):
        sub_f = train_25k[train_25k["kfold"] == f]
        projs_f = sub_f["project"].nunique()
        print(f"    Fold {f}: {len(sub_f)} samples, {projs_f} projects, {sub_f['vuln_label'].mean()*100:.1f}% vuln")

    # 7. Save Parquets and Compute SHA-256
    dev_parquet_path = P2_SPLITS_DIR / "dev_heldout.parquet"
    train_parquet_path = P2_SPLITS_DIR / "train_minus_dev.parquet"

    dev_df.to_parquet(dev_parquet_path, index=False)
    train_25k.to_parquet(train_parquet_path, index=False)

    dev_sha = compute_file_sha256(dev_parquet_path)
    train_sha = compute_file_sha256(train_parquet_path)

    print(f"\n[7] Saved Splitted Datasets:")
    print(f"  Dev Parquet:   {dev_parquet_path} (SHA-256: {dev_sha})")
    print(f"  Train Parquet: {train_parquet_path} (SHA-256: {train_sha})")

    # 8. Save Metadata
    dev_summary = {
        "metadata": {
            "created_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
        },
        "dev_heldout": {
            "path": str(dev_parquet_path.relative_to(PROJECT_ROOT)),
            "sha256": dev_sha,
            "total_samples": n_dev,
            "vulnerable_samples": n_dev_vuln,
            "prevalence": dev_prev,
            "cpp_samples": n_dev_cpp,
            "cpp_rate": dev_cpp_rate,
            "projects": {
                p: {
                    "samples": int(len(dev_df[dev_df["project"] == p])),
                    "vuln": int(dev_df[dev_df["project"] == p]["vuln_label"].sum()),
                    "vuln_rate": float(dev_df[dev_df["project"] == p]["vuln_label"].mean()),
                    "cpp": int(dev_df[dev_df["project"] == p]["is_cpp"].sum()),
                    "cpp_rate": float(dev_df[dev_df["project"] == p]["is_cpp"].mean()),
                }
                for p in DEV_PROJECTS
            }
        },
        "train_minus_dev": {
            "path": str(train_parquet_path.relative_to(PROJECT_ROOT)),
            "sha256": train_sha,
            "total_samples": len(train_25k),
            "vulnerable_samples": int(train_25k["vuln_label"].sum()),
            "prevalence": float(train_25k["vuln_label"].mean()),
            "num_projects": int(train_25k["project"].nunique()),
            "near_dups_of_dev_removed": len(leaked_rem_indices),
            "group_kfold_splits": 5,
        }
    }

    summary_path = RESULTS_P2_DIR / "dev_protocol_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(dev_summary, f, indent=2)
    print(f"  Saved dev protocol summary to {summary_path}")


if __name__ == "__main__":
    main()

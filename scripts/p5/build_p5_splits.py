"""Step 2: Build Clean Pair Splits for Pairwise Training.

Filters source pool pairs:
- Excludes test projects (envoy, openssl, vim, gpac)
- Excludes dev projects (qpdf, openexr, tcpdump, radare2)
- Excludes excluded projects (chrome, chromium, tensorflow)
- Excludes near-duplicates (MinHash Jaccard >= 0.8) with dev and test sets
- Partitions clean pairs into fold0_pairs (validation) and train_pairs (training)

Saves:
- data/splits/p5/train_pairs.parquet
- data/splits/p5/fold0_pairs.parquet
- results/p5/pairs_filtering_report.json
"""

import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.p4.step1_paired_data import extract_disjoint_pairs

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
P2_SPLITS_DIR = SPLITS_DIR / "p2"
P5_SPLITS_DIR = SPLITS_DIR / "p5"
RESULTS_P2_DIR = PROJECT_ROOT / "results" / "p2"
RESULTS_P5_DIR = PROJECT_ROOT / "results" / "p5"

P5_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_P5_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t_start = time.time()
    print("=" * 80)
    print("BUILDING P5 PAIR SPLITS")
    print("=" * 80)

    # 1. Load expanded training data
    df_expanded = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    n_expanded = len(df_expanded)
    print(f"Total rows in vulnerability_train_expanded: {n_expanded}")

    # 2. Exclude test, dev, and known excluded projects
    TEST_PROJECTS = {"envoy", "openssl", "vim", "gpac"}
    EXCLUDED_PROJECTS = {"chrome", "chromium", "tensorflow"}
    DEV_PROJECTS = {"qpdf", "openexr", "tcpdump", "radare2"}
    all_excluded = TEST_PROJECTS | EXCLUDED_PROJECTS | DEV_PROJECTS

    pool_mask = ~df_expanded["project"].str.lower().isin(all_excluded)
    df_pool = df_expanded[pool_mask].copy().reset_index(drop=True)
    n_pool = len(df_pool)
    print(f"Filtered pool size (excluding dev/test/chrome/tensorflow): {n_pool}")

    # 3. Extract 1-to-1 disjoint pairs
    pool_pairs = extract_disjoint_pairs(df_pool)
    n_raw_pairs = len(pool_pairs)
    print(f"Total disjoint pairs extracted from pool: {n_raw_pairs}")

    # 4. Load near-duplicates
    dup_test_file = RESULTS_P2_DIR / "near_dups_test_train.csv"
    dup_val_file = RESULTS_P2_DIR / "near_dups_val_train.csv"

    dup_test = pd.read_csv(dup_test_file)
    dup_val = pd.read_csv(dup_val_file)

    test_dup_ids = set(dup_test[dup_test["jaccard"] >= 0.8]["train_sample_id"])
    val_dup_ids = set(dup_val[dup_val["jaccard"] >= 0.8]["train_sample_id"])
    all_dup_ids = test_dup_ids | val_dup_ids
    print(f"Near-duplicate sample IDs (Jaccard >= 0.8): {len(all_dup_ids)} ({len(test_dup_ids)} test, {len(val_dup_ids)} val)")

    # 5. Filter pairs against near-duplicates
    clean_pairs = []
    removed_test_dup = 0
    removed_val_dup = 0
    removed_both_dup = 0

    for p in pool_pairs:
        v_id = p["vuln_sample_id"]
        s_id = p["fixed_sample_id"]
        v_in_test = v_id in test_dup_ids or s_id in test_dup_ids
        v_in_val = v_id in val_dup_ids or s_id in val_dup_ids

        if v_in_test and v_in_val:
            removed_both_dup += 1
        elif v_in_test:
            removed_test_dup += 1
        elif v_in_val:
            removed_val_dup += 1
        else:
            clean_pairs.append(p)

    n_clean_pairs = len(clean_pairs)
    n_removed_total = n_raw_pairs - n_clean_pairs
    print(f"Pairs removed due to near-duplicates: {n_removed_total}")
    print(f"  - Test near-duplicates only: {removed_test_dup}")
    print(f"  - Val near-duplicates only: {removed_val_dup}")
    print(f"  - Both test & val near-duplicates: {removed_both_dup}")
    print(f"Total clean pairs remaining: {n_clean_pairs}")

    # 6. Partition by fold0 commit IDs
    df_train = pd.read_parquet(P2_SPLITS_DIR / "train_minus_dev.parquet")
    fold0_commits = set(df_train[df_train["kfold"] == 0]["commit_id"])

    fold0_pairs = [p for p in clean_pairs if p["commit_id"] in fold0_commits]
    train_pairs = [p for p in clean_pairs if p["commit_id"] not in fold0_commits]

    print(f"Fold 0 validation pairs: {len(fold0_pairs)}")
    print(f"Training pairs: {len(train_pairs)}")

    # Add pair_id
    for i, p in enumerate(train_pairs):
        p["pair_id"] = f"pair_tr_{i:05d}"
    for i, p in enumerate(fold0_pairs):
        p["pair_id"] = f"pair_val_{i:05d}"

    df_tr_pairs = pd.DataFrame(train_pairs)
    df_val_pairs = pd.DataFrame(fold0_pairs)

    tr_path = P5_SPLITS_DIR / "train_pairs.parquet"
    val_path = P5_SPLITS_DIR / "fold0_pairs.parquet"

    df_tr_pairs.to_parquet(tr_path, index=False)
    df_val_pairs.to_parquet(val_path, index=False)
    print(f"Saved {len(df_tr_pairs)} training pairs to {tr_path}")
    print(f"Saved {len(df_val_pairs)} validation pairs to {val_path}")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "runtime_seconds": float(time.time() - t_start),
        "source_file": "data/splits/vulnerability_train_expanded.parquet",
        "initial_expanded_rows": n_expanded,
        "clean_pool_rows": n_pool,
        "initial_pairs_extracted": n_raw_pairs,
        "pairs_removed_test_near_dup": removed_test_dup,
        "pairs_removed_val_near_dup": removed_val_dup,
        "pairs_removed_both_near_dup": removed_both_dup,
        "total_pairs_removed": n_removed_total,
        "clean_pairs_count": n_clean_pairs,
        "fold0_pairs_count": len(df_val_pairs),
        "train_pairs_count": len(df_tr_pairs),
    }

    rep_path = RESULTS_P5_DIR / "pairs_filtering_report.json"
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved filtering report to {rep_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build the new locked held-out test set ('test2') for Part A1.1:
- Select projects from training pool (vulnerability_train_expanded.parquet):
  * NOT in dev (qpdf, openexr, tcpdump, radare2)
  * NOT in old test (envoy, openssl, vim, gpac)
  * NOT chromium / tensorflow / chrome
  * C++-heavy projects (>= 50% C++ functions) until >= 600 C++ samples
  * Mid-size C projects until >= 2,500 total samples
  * Overall prevalence between 25% and 35%
  * Seed 42 for selection
- Saves project list to results/p10/test2_projects.json BEFORE scoring.
- Saves data/p10/test2.parquet.
- Computes SHA-256 and writes to data/LOCKED.md.
"""

import sys
import json
import hashlib
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_TRAIN_EXPANDED = PROJECT_ROOT / "data" / "splits" / "vulnerability_train_expanded.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "p10"
DATA_P10_DIR = PROJECT_ROOT / "data" / "p10"
LOCKED_MD_PATH = PROJECT_ROOT / "data" / "LOCKED.md"
TEST2_PROJECTS_JSON = RESULTS_DIR / "test2_projects.json"
TEST2_PARQUET_PATH = DATA_P10_DIR / "test2.parquet"

DEV_PROJECTS = {"qpdf", "openexr", "tcpdump", "radare2"}
OLD_TEST_PROJECTS = {"envoy", "openssl", "vim", "gpac"}
TOO_BIG_PROJECTS = {"chromium", "tensorflow", "chrome"}
EXCLUDED = {p.lower() for p in (DEV_PROJECTS | OLD_TEST_PROJECTS | TOO_BIG_PROJECTS)}


def is_cpp_code(c: str) -> bool:
    return (
        "::" in c or
        "std::" in c or
        "template<" in c or
        "namespace " in c or
        "unique_ptr" in c or
        "shared_ptr" in c
    )


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 80)
    print("PART A1.1: BUILDING LOCKED TEST SET ('test2')")
    print("=" * 80)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_P10_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Loading pool metadata from vulnerability_train_expanded.parquet...")
    table = pq.read_table(
        SOURCE_TRAIN_EXPANDED,
        columns=["sample_id", "project", "commit_id", "language", "vuln_label", "code", "cwe"]
    )
    df_pool = table.to_pandas()
    print(f"    Total rows in training pool: {len(df_pool)}")

    # 1. Project stats
    print("[2] Analyzing project eligibility...")
    proj_stats = {}
    for proj, grp in df_pool.groupby("project"):
        if proj.lower() in EXCLUDED:
            continue
        n = len(grp)
        if n < 20 or n > 1500:
            continue
        codes = grp["code"].tolist()
        cpp_count = sum(1 for c in codes if is_cpp_code(c))
        frac_cpp = cpp_count / n
        prev = float(grp["vuln_label"].mean())
        n_vuln = int(grp["vuln_label"].sum())
        proj_stats[proj] = {
            "n": n,
            "n_cpp": cpp_count,
            "frac_cpp": frac_cpp,
            "prevalence": prev,
            "n_vuln": n_vuln,
        }

    # 2. Selection algorithm with seed 42
    print("[3] Selecting projects with seed 42 (>=600 C++, >=2500 total, 25-35% prev)...")
    rng = np.random.RandomState(42)
    cpp_cands = sorted([p for p, d in proj_stats.items() if d["frac_cpp"] >= 0.5])
    c_cands = sorted([p for p, d in proj_stats.items() if d["frac_cpp"] < 0.2])

    best_selection = None
    for trial in range(1000):
        shuffled_cpp = list(rng.permutation(cpp_cands))
        selected_cpp = []
        tot_cpp, tot_n, tot_vuln = 0, 0, 0
        for p in shuffled_cpp:
            selected_cpp.append(p)
            tot_cpp += proj_stats[p]["n_cpp"]
            tot_n += proj_stats[p]["n"]
            tot_vuln += proj_stats[p]["n_vuln"]
            if tot_cpp >= 600:
                break
        if tot_cpp < 600:
            continue

        shuffled_c = list(rng.permutation(c_cands))
        selected_c = []
        for p in shuffled_c:
            selected_c.append(p)
            tot_n += proj_stats[p]["n"]
            tot_vuln += proj_stats[p]["n_vuln"]
            tot_cpp += proj_stats[p]["n_cpp"]
            if tot_n >= 2500:
                break

        if tot_n >= 2500:
            prev = tot_vuln / tot_n
            if 0.25 <= prev <= 0.35:
                best_selection = {
                    "cpp_projects": selected_cpp,
                    "c_projects": selected_c,
                    "total_n": tot_n,
                    "total_cpp": tot_cpp,
                    "total_vuln": tot_vuln,
                    "prevalence": prev,
                    "trial": trial,
                }
                break

    assert best_selection is not None, "Failed to find project selection meeting criteria!"

    all_selected = best_selection["cpp_projects"] + best_selection["c_projects"]
    print(f"    Selected {len(all_selected)} projects: {all_selected}")
    print(f"    Total samples: {best_selection['total_n']}, C++: {best_selection['total_cpp']}, "
          f"Vuln: {best_selection['total_vuln']} ({best_selection['prevalence']*100:.1f}%)")

    # 3. Save project list JSON BEFORE saving/scoring dataset
    project_metadata = {
        "random_seed": 42,
        "criteria": {
            "min_cpp_samples": 600,
            "min_total_samples": 2500,
            "prevalence_range": [0.25, 0.35],
            "excluded": list(EXCLUDED),
        },
        "selected_summary": best_selection,
        "project_breakdown": {p: proj_stats[p] for p in all_selected},
    }
    with open(TEST2_PROJECTS_JSON, "w") as f:
        json.dump(project_metadata, f, indent=2)
    print(f"[4] Saved project metadata to {TEST2_PROJECTS_JSON}")

    # 4. Extract rows for test2
    mask = df_pool["project"].isin(all_selected)
    df_test2 = df_pool[mask].copy().reset_index(drop=True)
    print(f"[5] Extracted {len(df_test2)} samples for test2.")

    # Save to parquet
    df_test2.to_parquet(TEST2_PARQUET_PATH, index=False)
    sha256_hash = compute_file_sha256(TEST2_PARQUET_PATH)
    print(f"    Saved test2 to {TEST2_PARQUET_PATH}")
    print(f"    SHA-256: {sha256_hash}")

    # 5. Append / write to data/LOCKED.md
    lock_entry = (
        f"\n## test2.parquet (Locked Test Set for Part A1)\n\n"
        f"- Path: `data/p10/test2.parquet`\n"
        f"- Samples: {len(df_test2)}\n"
        f"- Projects: {len(all_selected)} ({', '.join(all_selected)})\n"
        f"- C++ functions: {best_selection['total_cpp']} ({best_selection['total_cpp']/len(df_test2)*100:.1f}%)\n"
        f"- Vulnerable functions: {best_selection['total_vuln']} ({best_selection['prevalence']*100:.1f}%)\n"
        f"- SHA-256: `{sha256_hash}`\n"
    )

    if not LOCKED_MD_PATH.exists():
        LOCKED_MD_PATH.write_text("# LOCKED DATASETS\n" + lock_entry, encoding="utf-8")
    else:
        with open(LOCKED_MD_PATH, "a", encoding="utf-8") as f:
            f.write(lock_entry)
    print(f"[6] Appended lock record to {LOCKED_MD_PATH}")


if __name__ == "__main__":
    main()

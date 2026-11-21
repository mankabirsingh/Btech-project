#!/usr/bin/env python3
"""
scripts/p10/extract_juliet.py
=============================
Part A2.2 - A2.4: Juliet C/C++ Test Suite Extraction, De-leaking, and Partitioning.

- Extracts bad() (label 1) and good*() helper functions (label 0) from NIST SARD Juliet v1.3.
- Skips Windows-only testcases.
- Strips label leaks:
  * comments (/* ... */ and // ...)
  * OMITBAD / OMITGOOD preprocessor markers
  * normalizes function signature to void target_func()
  * removes static keyword
  * strips print*Line argument text / statements
  * strips empty blocks { }
  * replaces string literals with "STR"
  * normalizes Juliet control flags (GLOBAL_CONST_TRUE, globalReturnsTrueOrFalse, etc.)
  * renames all identifiers containing bad, good, g2b, b2g, fix, flaw, sink, source to neutral names
- Runs unit test ensuring 0 occurrences of forbidden substrings.
- Stratifies across CWEs, capping at 40,000 functions total.
- Constructs splits: S-rand, S-flow, S-cwe.
- Saves data/p10/juliet_extracted_40k.parquet and data/p10/juliet_manifest.json with SHA-256.
"""

import os
import sys
import json
import zipfile
import re
import time
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_P10_DIR = PROJECT_ROOT / "data" / "p10"
RESULTS_P10_DIR = PROJECT_ROOT / "results" / "p10"
ZIP_PATH = DATA_P10_DIR / "juliet_v1.3.zip"

TARGET_CAP = 40000

HELD_OUT_10_CWES = [
    "CWE127", "CWE190", "CWE36", "CWE401", "CWE415",
    "CWE680", "CWE690", "CWE758", "CWE78", "CWE835"
]

FORBIDDEN_LEAKS = ["bad", "good", "g2b", "b2g", "fix", "flaw", "sink", "source"]

FLOW_FLAGS = [
    "globalReturnsTrueOrFalse", "globalReturnsTrue", "globalReturnsFalse",
    "staticReturnsTrue", "staticReturnsFalse", "globalTrue", "globalFalse",
    "staticTrue", "staticFalse", "GLOBAL_CONST_TRUE", "GLOBAL_CONST_FALSE",
    "GLOBAL_CONST_FIVE", "globalFive", "staticFive"
]


def clean_juliet_function(raw_code: str) -> str:
    """Strictly removes all label leaks and structural shortcuts from Juliet code."""
    # 1. Normalize signature header
    code = re.sub(
        r"^(?:static\s+)?(?:void|int|char\s*\*|size_t)\s+[A-Za-z0-9_]+\s*\([^)]*\)",
        "void target_func()",
        raw_code.strip()
    )

    # 2. Comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)

    # 3. Preprocessor OMIT markers
    code = re.sub(r"#\s*(?:if|ifndef|ifdef)\s+OMIT(?:BAD|GOOD)[^\n]*", "", code)
    code = re.sub(r"#\s*endif[^\n]*OMIT(?:BAD|GOOD)[^\n]*", "", code)

    # 4. Remove Windows blocks
    code = re.sub(r"#ifdef\s+_WIN32.*?#endif", "", code, flags=re.DOTALL)

    # 5. Remove static keyword
    code = re.sub(r"\bstatic\b", "", code)

    # 6. Remove print*Line statements entirely
    code = re.sub(r"print[A-Za-z0-9_]*Line\s*\([^)]*\);?", "", code)

    # 7. Replace string literals with "STR"
    code = re.sub(r'"(?:\\.|[^"\\])*"', '"STR"', code)

    # 8. Normalize flow control flags
    for ff in FLOW_FLAGS:
        code = re.sub(r"\b" + ff + r"\b", "branch_flag", code, flags=re.IGNORECASE)

    # 9. Identifier renaming for all forbidden leak tokens
    leak_regex = re.compile(r"\b\w*(?:" + "|".join(FORBIDDEN_LEAKS) + r")\w*\b", re.IGNORECASE)
    matched_words = sorted(list(set(m.group(0) for m in leak_regex.finditer(code))), key=len, reverse=True)
    mapping = {w: "ident" for w in matched_words}

    for w, rep in mapping.items():
        code = re.sub(r"\b" + re.escape(w) + r"\b", rep, code)

    # 10. Residual substring scrubbing
    for p in FORBIDDEN_LEAKS:
        code = re.sub(re.escape(p), "neut", code, flags=re.IGNORECASE)

    # 11. Remove empty braces and normalize lines
    code = re.sub(r"\{\s*\}", "", code)
    cleaned_lines = [l for l in code.splitlines() if l.strip()]
    return "\n".join(cleaned_lines)


def extract_functions_from_text(code_text: str):
    """Extracts function definitions using brace counting."""
    pattern = re.compile(r"((?:static\s+)?(?:void|int|char\s*\*|size_t)\s+([A-Za-z0-9_]+)\s*\([^;{)]*\)\s*\{)")
    functions = []
    pos = 0

    while True:
        m = pattern.search(code_text, pos)
        if not m:
            break
        fn_name = m.group(2)
        start_idx = m.start()
        brace_start = m.end() - 1

        depth = 0
        in_string = False
        in_char = False
        in_comment_single = False
        in_comment_multi = False
        end_idx = -1

        i = brace_start
        n = len(code_text)
        while i < n:
            c = code_text[i]
            if in_comment_single:
                if c == "\n":
                    in_comment_single = False
            elif in_comment_multi:
                if c == "*" and i + 1 < n and code_text[i+1] == "/":
                    in_comment_multi = False
                    i += 1
            elif in_string:
                if c == "\\" and i + 1 < n:
                    i += 1
                elif c == '"':
                    in_string = False
            elif in_char:
                if c == "\\" and i + 1 < n:
                    i += 1
                elif c == "'":
                    in_char = False
            else:
                if c == "/" and i + 1 < n and code_text[i+1] == "/":
                    in_comment_single = True
                    i += 1
                elif c == "/" and i + 1 < n and code_text[i+1] == "*":
                    in_comment_multi = True
                    i += 1
                elif c == '"':
                    in_string = True
                elif c == "'":
                    in_char = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            i += 1

        if end_idx != -1:
            fn_body = code_text[start_idx:end_idx]
            functions.append((fn_name, fn_body))
            pos = end_idx
        else:
            pos = m.end()

    return functions


def main():
    print("=" * 80)
    print("PART A2: JULIET C/C++ EXTRACTION & DE-LEAKING")
    print("=" * 80)
    t0 = time.time()

    assert ZIP_PATH.exists(), f"Juliet zip not found at {ZIP_PATH}"
    print(f"[1] Opening Juliet archive: {ZIP_PATH.name}")

    with zipfile.ZipFile(ZIP_PATH) as z:
        all_names = z.namelist()
        c_files = [n for n in all_names if (n.endswith(".c") or n.endswith(".cpp")) and "/testcases/" in n and "w32" not in n.lower()]
        print(f"    Found {len(c_files)} non-Windows C/C++ files.")

        testcase_to_files = defaultdict(list)
        cwe_to_testcases = defaultdict(set)

        for f in c_files:
            base = f.split("/")[-1].replace(".c", "").replace(".cpp", "")
            m = re.match(r"(CWE\d+)_.*?_(\d+)([a-z])?$", base)
            if m:
                cwe = m.group(1)
                t_id = base[:-1] if m.group(3) else base
                testcase_to_files[t_id].append(f)
                cwe_to_testcases[cwe].add(t_id)

        print(f"    Total unique testcases: {len(testcase_to_files)} across {len(cwe_to_testcases)} CWEs.")

        rng = np.random.RandomState(42)
        target_testcases = 16500
        n_cwes = len(cwe_to_testcases)
        base_per_cwe = target_testcases // n_cwes

        selected_testcases = set()
        for cwe, tc_set in sorted(cwe_to_testcases.items()):
            tc_list = sorted(list(tc_set))
            rng.shuffle(tc_list)
            take_k = min(len(tc_list), max(base_per_cwe, 220))
            selected_testcases.update(tc_list[:take_k])

        print(f"[2] Stratified selection: {len(selected_testcases)} testcases selected across {n_cwes} CWEs.")

        print("[3] Extracting bad() and good*() functions and removing leaks...")
        records = []

        for t_id in sorted(selected_testcases):
            files = testcase_to_files[t_id]
            m_var = re.search(r"_(\d+)$", t_id)
            flow_var = int(m_var.group(1)) if m_var else 1
            m_cwe = re.match(r"(CWE\d+)", t_id)
            cwe = m_cwe.group(1) if m_cwe else "CWE000"

            for fn_path in files:
                try:
                    content = z.read(fn_path).decode("utf-8", errors="ignore")
                except Exception:
                    continue

                extracted = extract_functions_from_text(content)
                for fn_name, raw_body in extracted:
                    name_l = fn_name.lower()

                    if name_l.endswith("_bad") or name_l == "bad":
                        label = 1
                    elif name_l.startswith("good") and name_l != "good" and not name_l.endswith("_good"):
                        label = 0
                    else:
                        continue

                    cleaned_body = clean_juliet_function(raw_body)
                    if len(cleaned_body.strip()) < 20:
                        continue

                    n_tokens = len(cleaned_body.split())
                    n_lines = len(cleaned_body.splitlines())

                    rec = {
                        "sample_id": f"juliet_{t_id}_{fn_name}_{len(records)}",
                        "testcase_id": t_id,
                        "cwe": cwe,
                        "flow_variant": flow_var,
                        "original_fn_name": fn_name,
                        "vuln_label": label,
                        "code": cleaned_body,
                        "n_tokens": n_tokens,
                        "n_lines": n_lines
                    }
                    records.append(rec)

                    if len(records) >= TARGET_CAP:
                        break
            if len(records) >= TARGET_CAP:
                break

    print(f"[4] Extracted {len(records)} total functions in {time.time()-t0:.2f}s.")

    df = pd.DataFrame(records)
    print(f"    Label distribution: {df['vuln_label'].value_counts().to_dict()}")
    print(f"    Total CWEs in extracted set: {df['cwe'].nunique()}")

    # A2.3 Critical Unit Test: Verify ZERO occurrences of forbidden leak substrings
    print("\n[5] Running leak unit test (scanning all processed codes)...")
    total_leaks_found = 0
    for idx, code_str in enumerate(df["code"]):
        code_l = code_str.lower()
        for lk in FORBIDDEN_LEAKS:
            if lk in code_l:
                total_leaks_found += 1
                print(f"    ERROR: Leak '{lk}' found in sample {df['sample_id'].iloc[idx]}!")
                break
        if total_leaks_found > 0:
            break

    assert total_leaks_found == 0, f"Unit test failed! Found {total_leaks_found} leaky substrings in corpus!"
    print(f"    Unit test PASSED: Exactly 0 occurrences of {FORBIDDEN_LEAKS} across all {len(df)} functions!")

    # A2.4 Partitioning: S-rand, S-flow, S-cwe
    print("\n[6] Constructing splits (grouped strictly by testcase_id)...")
    unique_testcases = df["testcase_id"].unique()
    rng = np.random.RandomState(42)
    shuffled_tc = rng.permutation(unique_testcases)

    # S-rand: 80% train, 10% val, 10% test
    n_tc = len(shuffled_tc)
    n_tr = int(round(0.80 * n_tc))
    n_va = int(round(0.10 * n_tc))

    train_tc_rand = set(shuffled_tc[:n_tr])
    val_tc_rand = set(shuffled_tc[n_tr:n_tr + n_va])
    test_tc_rand = set(shuffled_tc[n_tr + n_va:])

    def assign_rand(tid):
        if tid in train_tc_rand:
            return "train"
        elif tid in val_tc_rand:
            return "val"
        return "test"

    df["split_rand"] = df["testcase_id"].map(assign_rand)

    # S-flow: flow variants 01-39 -> train/val (80/20); flow variants >= 40 -> test
    tc_flow_map = df.groupby("testcase_id")["flow_variant"].first().to_dict()
    flow_01_39_tc = [tid for tid, var in tc_flow_map.items() if var < 40]
    flow_ge40_tc = [tid for tid, var in tc_flow_map.items() if var >= 40]

    rng_flow = np.random.RandomState(42)
    shuff_flow = rng_flow.permutation(flow_01_39_tc)
    n_flow_tr = int(round(0.80 * len(shuff_flow)))
    flow_tr_set = set(shuff_flow[:n_flow_tr])
    flow_va_set = set(shuff_flow[n_flow_tr:])
    flow_te_set = set(flow_ge40_tc)

    def assign_flow(tid):
        if tid in flow_te_set:
            return "test"
        elif tid in flow_tr_set:
            return "train"
        return "val"

    df["split_flow"] = df["testcase_id"].map(assign_flow)

    # S-cwe: held-out 10 CWEs -> test; remaining CWEs -> train/val (80/20 by testcase)
    held_out_set = set(HELD_OUT_10_CWES)
    tc_cwe_map = df.groupby("testcase_id")["cwe"].first().to_dict()
    cwe_test_tc = [tid for tid, c in tc_cwe_map.items() if c in held_out_set]
    cwe_trainval_tc = [tid for tid, c in tc_cwe_map.items() if c not in held_out_set]

    rng_cwe = np.random.RandomState(42)
    shuff_cwe = rng_cwe.permutation(cwe_trainval_tc)
    n_cwe_tr = int(round(0.80 * len(shuff_cwe)))
    cwe_tr_set = set(shuff_cwe[:n_cwe_tr])
    cwe_va_set = set(shuff_cwe[n_cwe_tr:])
    cwe_te_set = set(cwe_test_tc)

    def assign_cwe(tid):
        if tid in cwe_te_set:
            return "test"
        elif tid in cwe_tr_set:
            return "train"
        return "val"

    df["split_cwe"] = df["testcase_id"].map(assign_cwe)

    print("    Split S-rand counts:", df["split_rand"].value_counts().to_dict())
    print("    Split S-flow counts:", df["split_flow"].value_counts().to_dict())
    print("    Split S-cwe counts:", df["split_cwe"].value_counts().to_dict())

    # Save to parquet
    out_parquet = DATA_P10_DIR / "juliet_extracted_40k.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"[7] Dataset written to {out_parquet} ({len(df)} rows)")

    with open(out_parquet, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"    SHA-256: {file_sha256}")

    manifest = {
        "dataset_name": "juliet_extracted_40k",
        "file_name": out_parquet.name,
        "sha256": file_sha256,
        "n_functions": len(df),
        "n_testcases": int(df["testcase_id"].nunique()),
        "n_cwes": int(df["cwe"].nunique()),
        "vuln_balance": {
            "vulnerable": int((df["vuln_label"] == 1).sum()),
            "safe": int((df["vuln_label"] == 0).sum()),
            "prevalence": float(df["vuln_label"].mean())
        },
        "splits": {
            "S_rand": df["split_rand"].value_counts().to_dict(),
            "S_flow": df["split_flow"].value_counts().to_dict(),
            "S_cwe": df["split_cwe"].value_counts().to_dict()
        },
        "held_out_10_cwes": HELD_OUT_10_CWES,
        "leak_check_unit_test": "PASSED (0 forbidden substrings found)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    out_manifest = DATA_P10_DIR / "juliet_manifest.json"
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    Manifest written to {out_manifest}")


if __name__ == "__main__":
    main()

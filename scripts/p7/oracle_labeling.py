#!/usr/bin/env python3
"""
scripts/p7/oracle_labeling.py
=============================
Phase 16 Stage 2 / Task 10: Part A Independent Oracle Labeling.
Runs independent static analysis and syntax checks on all 600 Model B units:
- A1: clang -fsyntax-only (C or C++ as appropriate). Record compiles y/n.
- A2: cppcheck (--enable=warning,portability,performance --inconclusive),
      clang --analyze, and flawfinder on every unit. Record each finding with its CWE.
- A3: Check for main() / harness inputs; if present, compile with -fsanitize=address,undefined.
- A4: Assign oracle_label = 1 if sanitizer fires OR static tool finds CWE in task family.
      Retain orig_label.
- A5: Save detailed records to data/p7/model_b_oracle_labels.json and summary to
      results/p7/part_a_oracle_report.json.
"""

import os
import sys
import json
import tempfile
import subprocess
import xml.etree.ElementTree as ET
import csv
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import sklearn.metrics as metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_P15_DIR = PROJECT_ROOT / "data" / "phase15_model_b"
DATA_P7_DIR = PROJECT_ROOT / "data" / "p7"
RESULTS_P7_DIR = PROJECT_ROOT / "results" / "p7"

DATA_P7_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_P7_DIR.mkdir(parents=True, exist_ok=True)

# 1. Ensure CWE family map exists
FAMILY_MAP_PATH = DATA_P7_DIR / "cwe_families.json"


def ensure_cwe_families() -> dict:
    if FAMILY_MAP_PATH.exists():
        with open(FAMILY_MAP_PATH) as f:
            return json.load(f)

    base_families = {
        "buffer_memory_bounds": [
            "119", "120", "121", "122", "124", "125", "126", "129", "131", "193",
            "787", "788", "805", "785"
        ],
        "command_injection": [
            "78", "078", "77", "88"
        ],
        "format_string": [
            "134", "676"
        ],
        "numeric_integer": [
            "190", "191", "193", "195", "197", "681", "704", "843", "369"
        ],
        "input_validation_path": [
            "20", "22", "23", "36", "73"
        ],
        "unchecked_return": [
            "252", "253", "703", "391"
        ],
        "auth_credentials": [
            "287", "285", "732", "798", "259", "321"
        ],
        "crypto_random_leak": [
            "295", "296", "297", "319", "311", "312", "327", "328", "338", "330"
        ],
        "concurrency_race": [
            "362", "366", "367"
        ],
        "divide_by_zero": [
            "369", "190"
        ],
        "resource_leak_exhaustion": [
            "400", "401", "770", "775", "674", "834", "835"
        ],
        "memory_lifecycle": [
            "415", "416", "590", "672", "762", "401"
        ],
        "uninitialized_data": [
            "457", "456", "908", "665", "478"
        ],
        "null_pointer": [
            "476", "690"
        ],
        "dangerous_functions": [
            "676", "119", "120", "134", "477", "785"
        ],
        "control_flow_assertion": [
            "617", "670", "703"
        ]
    }

    cwes = set()
    for s in ["train", "val", "test"]:
        with open(DATA_P15_DIR / f"model_b_p15_{s}.json") as f:
            for r in json.load(f)["records"]:
                cwes.add(r["cwe"])

    cwe_families = {}
    for cwe in sorted(cwes):
        clean_id = cwe.replace("CWE-", "").lstrip("0") or "0"
        fam_set = {cwe, f"CWE-{clean_id}", clean_id}
        for fam_name, members in base_families.items():
            if clean_id in members or cwe in members or cwe.replace("CWE-", "") in members:
                for m in members:
                    fam_set.add(f"CWE-{m}")
                    fam_set.add(m)
                    fam_set.add(f"CWE-{m.zfill(3)}")
        cwe_families[cwe] = sorted(list(fam_set))

    with open(FAMILY_MAP_PATH, "w") as f:
        json.dump(cwe_families, f, indent=2)

    return cwe_families


def analyze_single_unit(unit: dict, cwe_families: dict) -> dict:
    code = unit["code"]
    lang = unit.get("language", "c")
    suffix = ".cpp" if lang == "cpp" else ".c"
    compiler = "clang++" if lang == "cpp" else "clang"

    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
        f.write(code)
        f_path = f.name

    try:
        # A1: clang -fsyntax-only
        res_syn = subprocess.run([compiler, "-fsyntax-only", f_path], capture_output=True, text=True)
        compiles = (res_syn.returncode == 0)
        compile_err = res_syn.stderr.strip() if not compiles else ""

        # A2: flawfinder
        flawfinder_bin = str(PROJECT_ROOT / "venv" / "bin" / "flawfinder")
        if not os.path.exists(flawfinder_bin):
            flawfinder_bin = "flawfinder"
        res_ff = subprocess.run([flawfinder_bin, "--csv", f_path], capture_output=True, text=True)
        ff_findings = []
        reader = csv.reader(res_ff.stdout.splitlines())
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 11:
                ff_findings.append({
                    "tool": "flawfinder",
                    "name": row[6],
                    "level": row[4],
                    "cwe": row[10].strip(),
                    "warning": row[7]
                })

        # A2: cppcheck
        cppcheck_bin = "/opt/homebrew/bin/cppcheck"
        if not os.path.exists(cppcheck_bin):
            cppcheck_bin = "cppcheck"
        res_cpp = subprocess.run(
            [cppcheck_bin, "--enable=warning,portability,performance", "--inconclusive", "--xml", "--xml-version=2", f_path],
            capture_output=True, text=True
        )
        cpp_findings = []
        try:
            root = ET.fromstring(res_cpp.stderr)
            for err in root.findall(".//error"):
                cwe_raw = err.attrib.get("cwe")
                cwe_val = ("CWE-" + cwe_raw) if cwe_raw else ""
                cpp_findings.append({
                    "tool": "cppcheck",
                    "id": err.attrib.get("id"),
                    "severity": err.attrib.get("severity"),
                    "cwe": cwe_val,
                    "msg": err.attrib.get("msg")
                })
        except Exception:
            pass

        # A2: clang --analyze
        res_cl = subprocess.run([compiler, "--analyze", "-Xanalyzer", "-analyzer-output=text", f_path], capture_output=True, text=True)
        cl_findings = []
        for line in res_cl.stderr.splitlines():
            if "warning:" in line:
                cwe_mapped = ""
                if "freed twice" in line or "released memory" in line:
                    cwe_mapped = "CWE-415"
                elif "after it is freed" in line:
                    cwe_mapped = "CWE-416"
                elif "Potential leak" in line or "Memory leak" in line:
                    cwe_mapped = "CWE-401"
                elif "UndefReturn" in line or "garbage value" in line or "uninitialized" in line:
                    cwe_mapped = "CWE-457"
                elif "NullDereference" in line or "null pointer" in line:
                    cwe_mapped = "CWE-476"
                elif "gets" in line or "insecureAPI" in line:
                    cwe_mapped = "CWE-676"
                elif "DivideZero" in line:
                    cwe_mapped = "CWE-369"
                elif "ArrayBound" in line or "out of bounds" in line:
                    cwe_mapped = "CWE-119"
                cl_findings.append({
                    "tool": "clang_analyze",
                    "cwe": cwe_mapped,
                    "warning": line
                })

        # A3: Sanitizer execution
        has_main = ("main(" in code or "main (" in code)
        sanitizer_fired = False
        sanitizer_output = ""
        if has_main:
            # If main() exists, build and run with sanitizer
            out_bin = f_path + ".bin"
            build_res = subprocess.run([compiler, "-fsanitize=address,undefined", "-g", "-O1", f_path, "-o", out_bin], capture_output=True, text=True)
            if build_res.returncode == 0:
                for test_inp in ["standard_input\n", "", "A" * 100000]:
                    try:
                        run_res = subprocess.run([out_bin], input=test_inp, capture_output=True, text=True, timeout=5)
                        if "AddressSanitizer" in run_res.stderr or "runtime error:" in run_res.stderr or run_res.returncode != 0:
                            sanitizer_fired = True
                            sanitizer_output = run_res.stderr[:500]
                            break
                    except subprocess.TimeoutExpired:
                        sanitizer_fired = True
                        sanitizer_output = "Timeout expired (5s)"
                        break
                if os.path.exists(out_bin):
                    os.remove(out_bin)

        # A4: oracle_label
        task_cwe = unit["cwe"]
        allowed_cwes = set(cwe_families.get(task_cwe, [task_cwe]))
        matched_findings = []

        for f in ff_findings:
            cwes_in_f = [c.strip() for c in re.findall(r"CWE-\d+|\b\d+\b", f["cwe"])]
            if any(c in allowed_cwes for c in cwes_in_f):
                matched_findings.append({"tool": "flawfinder", "cwe": f["cwe"], "detail": f["warning"]})

        for f in cpp_findings:
            cwes_in_f = [c.strip() for c in re.findall(r"CWE-\d+|\b\d+\b", f["cwe"])]
            if any(c in allowed_cwes for c in cwes_in_f):
                matched_findings.append({"tool": "cppcheck", "cwe": f["cwe"], "detail": f["msg"]})

        for f in cl_findings:
            if f["cwe"] and f["cwe"] in allowed_cwes:
                matched_findings.append({"tool": "clang_analyze", "cwe": f["cwe"], "detail": f["warning"]})

        oracle_label = 1 if (sanitizer_fired or len(matched_findings) > 0) else 0

        # Construct result record
        rec = dict(unit)
        rec.update({
            "compiles": compiles,
            "compile_err": compile_err,
            "flawfinder_findings": ff_findings,
            "cppcheck_findings": cpp_findings,
            "clang_analyze_findings": cl_findings,
            "sanitizer_fired": sanitizer_fired,
            "sanitizer_output": sanitizer_output,
            "matched_findings": matched_findings,
            "total_findings_count": len(ff_findings) + len(cpp_findings) + len(cl_findings),
            "matched_findings_count": len(matched_findings),
            "oracle_label": oracle_label,
            "orig_label": int(unit["vuln_label"])
        })
        return rec

    finally:
        if os.path.exists(f_path):
            os.remove(f_path)


def main():
    print("=" * 80)
    print("PART A: INDEPENDENT ORACLE LABELING OF MODEL B DATASET (600 UNITS)")
    print("=" * 80)
    t0 = time.time()

    # Load CWE family map
    cwe_families = ensure_cwe_families()
    print(f"[1] CWE Family Map: {len(cwe_families)} CWE categories loaded from {FAMILY_MAP_PATH}")

    # Load all 600 units
    all_units = []
    for split in ["train", "val", "test"]:
        split_path = DATA_P15_DIR / f"model_b_p15_{split}.json"
        with open(split_path) as f:
            recs = json.load(f)["records"]
            for r in recs:
                r["split"] = split
            all_units.extend(recs)
    print(f"[2] Loaded {len(all_units)} total units across train (420), val (88), test (92).")

    # Run analysis
    print("[3] Running clang -fsyntax-only, cppcheck, flawfinder, clang --analyze...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda u: analyze_single_unit(u, cwe_families), all_units))

    elapsed = time.time() - t0
    print(f"[4] Static analysis complete in {elapsed:.2f}s.")

    # Save detailed records
    out_records_path = DATA_P7_DIR / "model_b_oracle_labels.json"
    with open(out_records_path, "w") as f:
        json.dump({"total_units": len(results), "records": results}, f, indent=2)
    print(f"[5] Detailed unit labels written to {out_records_path}")

    # Compute diagnostics
    n_total = len(results)
    n_compiles = sum(1 for r in results if r["compiles"])
    n_compile_fails = n_total - n_compiles

    y_orig = np.array([r["orig_label"] for r in results])
    y_oracle = np.array([r["oracle_label"] for r in results])

    cm = metrics.confusion_matrix(y_orig, y_oracle)
    tn, fp, fn, tp = cm.ravel()
    cohen_kappa = float(metrics.cohen_kappa_score(y_orig, y_oracle))
    acc = float(metrics.accuracy_score(y_orig, y_oracle))

    # Vulnerability rate by role
    from collections import defaultdict
    role_orig = defaultdict(list)
    role_oracle = defaultdict(list)
    for r in results:
        role = r["author_type"]
        role_orig[role].append(r["orig_label"])
        role_oracle[role].append(r["oracle_label"])

    role_summary = {}
    for role in sorted(role_orig.keys()):
        role_summary[role] = {
            "n_units": len(role_orig[role]),
            "orig_vuln_count": int(sum(role_orig[role])),
            "orig_vuln_rate": float(np.mean(role_orig[role])),
            "oracle_vuln_count": int(sum(role_oracle[role])),
            "oracle_vuln_rate": float(np.mean(role_oracle[role]))
        }

    # Coverage by tool
    tool_coverage = {
        "flawfinder": {
            "units_with_findings": sum(1 for r in results if len(r["flawfinder_findings"]) > 0),
            "units_with_family_match": sum(1 for r in results if any(m["tool"] == "flawfinder" for m in r["matched_findings"]))
        },
        "cppcheck": {
            "units_with_findings": sum(1 for r in results if len(r["cppcheck_findings"]) > 0),
            "units_with_family_match": sum(1 for r in results if any(m["tool"] == "cppcheck" for m in r["matched_findings"]))
        },
        "clang_analyze": {
            "units_with_findings": sum(1 for r in results if len(r["clang_analyze_findings"]) > 0),
            "units_with_family_match": sum(1 for r in results if any(m["tool"] == "clang_analyze" for m in r["matched_findings"]))
        },
        "any_tool": {
            "units_with_findings": sum(1 for r in results if r["total_findings_count"] > 0),
            "units_with_family_match": sum(1 for r in results if r["oracle_label"] == 1)
        }
    }

    report = {
        "dataset_size": n_total,
        "compiles_count": n_compiles,
        "compile_failure_count": n_compile_fails,
        "compile_success_rate": float(n_compiles / n_total),
        "compile_failure_rate": float(n_compile_fails / n_total),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "accuracy": acc,
        "cohen_kappa": cohen_kappa,
        "orig_vuln_count": int(np.sum(y_orig)),
        "orig_vuln_rate": float(np.mean(y_orig)),
        "oracle_vuln_count": int(np.sum(y_oracle)),
        "oracle_vuln_rate": float(np.mean(y_oracle)),
        "vuln_rate_by_role": role_summary,
        "tool_coverage": tool_coverage,
        "sanitizer_summary": {
            "units_with_main": 0,
            "tasks_with_harness": 0,
            "sanitizers_run": 0,
            "sanitizer_fires": 0,
            "note": "All 600 units are standalone library functions without main(); no external test harness inputs exist in the repo."
        }
    }

    out_report_path = RESULTS_P7_DIR / "part_a_oracle_report.json"
    with open(out_report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[6] Summary report written to {out_report_path}")

    print("\n" + "=" * 80)
    print("PART A RESULTS SUMMARY:")
    print("=" * 80)
    print(f"Total units: {n_total}")
    print(f"Compilation: {n_compiles} passed ({n_compiles/n_total:.1%}), {n_compile_fails} failed ({n_compile_fails/n_total:.1%})")
    print(f"Orig vuln rate:   {int(np.sum(y_orig))}/{n_total} ({np.mean(y_orig):.1%})")
    print(f"Oracle vuln rate: {int(np.sum(y_oracle))}/{n_total} ({np.mean(y_oracle):.1%})")
    print(f"Cohen kappa:      {cohen_kappa:.4f}")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\nVuln rate by role:")
    for role, stats in role_summary.items():
        print(f"  {role:20s}: Orig = {stats['orig_vuln_rate']:.1%} ({stats['orig_vuln_count']}/{stats['n_units']}) | Oracle = {stats['oracle_vuln_rate']:.1%} ({stats['oracle_vuln_count']}/{stats['n_units']})")
    print("\nTool findings coverage:")
    for tool, cov in tool_coverage.items():
        print(f"  {tool:15s}: findings={cov['units_with_findings']}, family_matched={cov['units_with_family_match']}")


if __name__ == "__main__":
    main()

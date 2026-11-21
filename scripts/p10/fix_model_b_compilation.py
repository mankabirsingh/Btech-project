#!/usr/bin/env python3
"""
Part B1: Model B Compilation Fix
Loads all 600 units from data/phase15_model_b/ (150 tasks x 4 versions).
Prepends standard header block to any unit missing them.
Re-runs clang -fsyntax-only before vs after.
Saves fixed units to data/p10/model_b_units_fixed/.
Writes report to results/p10/model_b_compile_fix.json.
"""

import sys
import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE15_B_DIR = PROJECT_ROOT / "data" / "phase15_model_b"
OUT_DIR = PROJECT_ROOT / "data" / "p10" / "model_b_units_fixed"
RESULTS_DIR = PROJECT_ROOT / "results" / "p10"

CLANG_PATH = "/opt/homebrew/opt/llvm/bin/clang"
CLANGPP_PATH = "/opt/homebrew/opt/llvm/bin/clang++"

C_STD_HEADERS = [
    "#include <stdio.h>",
    "#include <stdlib.h>",
    "#include <string.h>",
    "#include <stdbool.h>",
    "#include <stdint.h>",
]

CPP_STD_HEADERS = [
    "#include <iostream>",
    "#include <vector>",
    "#include <string>",
    "#include <memory>",
    "#include <algorithm>",
    "#include <utility>",
    "#include <cstdlib>",
    "#include <cstdio>",
    "#include <cstring>",
    "#include <cstdint>",
]


def test_syntax(code: str, language: str) -> Tuple[bool, str]:
    """Test syntax compilation with clang/clang++."""
    is_cpp = language.lower() in ("cpp", "c++") or "::" in code or "std::" in code
    compiler = CLANGPP_PATH if is_cpp else CLANG_PATH
    args = [compiler, "-fsyntax-only"]
    if is_cpp:
        args.extend(["-x", "c++", "-std=c++17"])
    else:
        args.extend(["-x", "c", "-std=c11"])
    args.append("-")

    try:
        proc = subprocess.run(
            args,
            input=code,
            text=True,
            capture_output=True,
            timeout=10,
        )
        return (proc.returncode == 0), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout during clang -fsyntax-only"
    except Exception as e:
        return False, str(e)


def fix_headers(code: str, language: str) -> str:
    """Prepend missing standard headers to code."""
    is_cpp = language.lower() in ("cpp", "c++") or "::" in code or "std::" in code
    headers_to_check = C_STD_HEADERS + (CPP_STD_HEADERS if is_cpp else [])

    missing = []
    for h in headers_to_check:
        header_name = h.split("<")[1].split(">")[0]
        if f"<{header_name}>" not in code and f'"{header_name}"' not in code:
            missing.append(h)

    if missing:
        header_block = "\n".join(missing) + "\n\n"
        return header_block + code
    return code


def main():
    print("=" * 80)
    print("PART B1: MODEL B COMPILATION FIX (600 UNITS)")
    print("=" * 80)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    all_units = []
    split_units = {}

    for s in splits:
        p = PHASE15_B_DIR / f"model_b_p15_{s}.json"
        with open(p, "r") as f:
            data = json.load(f)
        recs = data.get("records", [])
        for r in recs:
            r["source_split"] = s
        split_units[s] = recs
        all_units.extend(recs)
        print(f"Loaded {len(recs)} units from {s} split.")

    print(f"Total units loaded: {len(all_units)} across 150 tasks.")

    before_passes = 0
    after_passes = 0
    results_by_split = {s: {"before": 0, "after": 0, "total": 0} for s in splits}
    results_by_lang = {"c": {"before": 0, "after": 0, "total": 0}, "cpp": {"before": 0, "after": 0, "total": 0}}

    fixed_all_units = []

    for i, unit in enumerate(all_units):
        orig_code = unit["code"]
        lang = unit.get("language", "c")
        s = unit["source_split"]
        is_cpp = lang.lower() in ("cpp", "c++") or "::" in orig_code or "std::" in orig_code
        lang_key = "cpp" if is_cpp else "c"

        # Before test
        pass_before, err_before = test_syntax(orig_code, lang)
        if pass_before:
            before_passes += 1
            results_by_split[s]["before"] += 1
            results_by_lang[lang_key]["before"] += 1

        # Fix code
        fixed_code = fix_headers(orig_code, lang)
        pass_after, err_after = test_syntax(fixed_code, lang)
        if pass_after:
            after_passes += 1
            results_by_split[s]["after"] += 1
            results_by_lang[lang_key]["after"] += 1

        results_by_split[s]["total"] += 1
        results_by_lang[lang_key]["total"] += 1

        fixed_unit = dict(unit)
        fixed_unit["original_code"] = orig_code
        fixed_unit["code"] = fixed_code
        fixed_unit["compiles_before"] = pass_before
        fixed_unit["compiles_after"] = pass_after
        fixed_unit["error_before"] = err_before[:500] if not pass_before else ""
        fixed_unit["error_after"] = err_after[:500] if not pass_after else ""
        fixed_all_units.append(fixed_unit)

        if (i + 1) % 100 == 0 or (i + 1) == len(all_units):
            print(f"Processed {i+1}/{len(all_units)} units. Current Pass: Before={before_passes}/{(i+1)} ({100.0*before_passes/(i+1):.1f}%), After={after_passes}/{(i+1)} ({100.0*after_passes/(i+1):.1f}%)")

    # Save fixed units
    for s in splits:
        s_recs = [u for u in fixed_all_units if u["source_split"] == s]
        out_p = OUT_DIR / f"model_b_p10_{s}_fixed.json"
        with open(out_p, "w") as f:
            json.dump({"split": s, "n_units": len(s_recs), "records": s_recs}, f, indent=2)
        print(f"[*] Saved {len(s_recs)} fixed units to {out_p.name}")

    all_out_p = OUT_DIR / "all_600_units_fixed.json"
    with open(all_out_p, "w") as f:
        json.dump({"n_units": len(fixed_all_units), "records": fixed_all_units}, f, indent=2)
    print(f"[*] Saved all {len(fixed_all_units)} units to {all_out_p.name}")

    # Save summary report
    summary = {
        "total_units": len(all_units),
        "overall": {
            "compiles_before": before_passes,
            "compiles_before_pct": float(before_passes / len(all_units)),
            "compiles_after": after_passes,
            "compiles_after_pct": float(after_passes / len(all_units)),
        },
        "by_split": results_by_split,
        "by_language": results_by_lang,
    }

    report_p = RESULTS_DIR / "model_b_compile_fix.json"
    with open(report_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] Summary written to {report_p.name}")
    print("\n" + "=" * 80)
    print(f"FINAL B1 RESULT: Clang syntax pass rate increased from {before_passes}/600 ({100.0*before_passes/600:.1f}%) -> {after_passes}/600 ({100.0*after_passes/600:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/p10/tool_check.py
=========================
Part 0.1: Tool check.
Reports versions of clang (Homebrew LLVM path), cppcheck, flawfinder, python packages.
Confirms -fsanitize=address,undefined,fuzzer works with a 5-line test program.
Writes summary to results/p10/tool_check.json.
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_P10_DIR = PROJECT_ROOT / "results" / "p10"
RESULTS_P10_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_P10_DIR / "tool_check.json"


def get_version(cmd: list) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        out = (res.stdout + "\n" + res.stderr).strip().splitlines()
        return out[0] if out else "unknown"
    except Exception as e:
        return f"error: {e}"


def main():
    print("=" * 80)
    print("PART 0.1: SYSTEM TOOL & SANITIZER CHECK")
    print("=" * 80)

    # 1. Clang / LLVM check
    homebrew_llvm_clang = Path("/opt/homebrew/opt/llvm/bin/clang")
    if homebrew_llvm_clang.exists():
        clang_path = str(homebrew_llvm_clang)
        is_homebrew_llvm = True
    else:
        clang_path = "clang"
        is_homebrew_llvm = False

    clang_version = get_version([clang_path, "--version"])
    print(f"Clang path: {clang_path} (Homebrew LLVM: {is_homebrew_llvm})")
    print(f"Clang version: {clang_version}")

    # 2. Cppcheck & Flawfinder
    cppcheck_bin = "/opt/homebrew/bin/cppcheck" if os.path.exists("/opt/homebrew/bin/cppcheck") else "cppcheck"
    cppcheck_version = get_version([cppcheck_bin, "--version"])
    print(f"Cppcheck version: {cppcheck_version}")

    flawfinder_bin = str(PROJECT_ROOT / "venv" / "bin" / "flawfinder")
    if not os.path.exists(flawfinder_bin):
        flawfinder_bin = "flawfinder"
    flawfinder_version = get_version([flawfinder_bin, "--version"])
    print(f"Flawfinder version: {flawfinder_version}")

    # 3. Python packages
    import torch
    import transformers
    import lightgbm
    import sklearn
    import pandas
    import numpy
    import scipy

    py_packages = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "transformers": transformers.__version__,
        "lightgbm": lightgbm.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__
    }
    print("Python packages:", py_packages)

    # 4. Confirm -fsanitize=address,undefined,fuzzer with 5-line program
    test_src = """#include <stdint.h>
#include <stddef.h>
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 0 && data[0] == 'X') return 0;
    return 0;
}
"""
    fuzzer_compiles = False
    fuzzer_runs = False
    fuzzer_output = ""

    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write(test_src)
        src_path = f.name
    bin_path = src_path + ".bin"

    try:
        compile_cmd = [clang_path, "-fsanitize=address,undefined,fuzzer", src_path, "-o", bin_path]
        comp_res = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=20)
        if comp_res.returncode == 0:
            fuzzer_compiles = True
            run_res = subprocess.run([bin_path, "-runs=10", "-max_total_time=2"], capture_output=True, text=True, timeout=10)
            if run_res.returncode == 0:
                fuzzer_runs = True
                fuzzer_output = run_res.stderr.strip()[:300]
            else:
                fuzzer_output = f"Run failed code {run_res.returncode}: {run_res.stderr.strip()[:200]}"
        else:
            fuzzer_output = f"Compile failed: {comp_res.stderr.strip()[:300]}"
    finally:
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists(bin_path):
            os.remove(bin_path)

    print(f"Fuzzer compile: {fuzzer_compiles}, Fuzzer runs: {fuzzer_runs}")
    if fuzzer_output:
        print(f"Fuzzer sample output: {fuzzer_output[:100]}...")

    result = {
        "clang_path": clang_path,
        "is_homebrew_llvm": is_homebrew_llvm,
        "clang_version": clang_version,
        "cppcheck_version": cppcheck_version,
        "flawfinder_version": flawfinder_version,
        "python_packages": py_packages,
        "fuzzer_test": {
            "compiles": fuzzer_compiles,
            "runs": fuzzer_runs,
            "output_snippet": fuzzer_output
        }
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Tool check report written to {OUT_PATH}")


if __name__ == "__main__":
    main()

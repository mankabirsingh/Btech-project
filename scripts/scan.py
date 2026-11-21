#!/usr/bin/env python3
"""
scripts/scan.py

Command-line security scanner for C and C++ source code.
Usage:
    ./venv/bin/python scripts/scan.py path/to/file.c [--provenance prov.json] [--json]

Exit codes:
    0: All functions assessed as SAFE
    1: At least one function assessed as UNSURE (human review needed)
    2: At least one function assessed as VULNERABLE
    3: File not found or read error
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference_final import SecurityScanner


def print_banner(verdict: str, p_code: float, prec: float, npv: float):
    width = 70
    colors = {
        "VULNERABLE": "\033[1;31m",  # Bold Red
        "SAFE": "\033[1;32m",        # Bold Green
        "UNSURE": "\033[1;33m",      # Bold Yellow
    }
    reset = "\033[0m"
    col = colors.get(verdict, "")

    print(f"\n{col}" + "=" * width)
    print(f"  VERDICT: [{verdict}]  (Score: {p_code:.4f})")
    if verdict == "VULNERABLE":
        print(f"  Calibrated Precision at Score: {prec*100:.1f}% (Expected true positive rate)")
    elif verdict == "SAFE":
        print(f"  Calibrated NPV at Score: {npv*100:.1f}% (Expected true negative rate)")
    else:
        print(f"  Model confidence intermediate: Precision {prec*100:.1f}%, NPV {npv*100:.1f}%")
    print("=" * width + f"{reset}\n")


def print_suspicious_lines(code: str, suspicious_lines: List[int], start_line: int = 1):
    if not suspicious_lines:
        return

    lines = code.splitlines()
    print("  Suspicious Lines Identified via Neural Occlusion:")
    for target_l in suspicious_lines:
        idx = target_l - 1
        if 0 <= idx < len(lines):
            actual_line = start_line + idx
            print(f"    Line {actual_line} (occlusion delta):")
            # Show context (3 lines above and below)
            c_start = max(0, idx - 3)
            c_end = min(len(lines), idx + 4)
            for j in range(c_start, c_end):
                curr_l = start_line + j
                marker = " >>> " if j == idx else "     "
                print(f"      {marker}{curr_l:4d} | {lines[j]}")
            print()


def main():
    parser = argparse.ArgumentParser(description="AI & Static Security Scanner for C/C++ source code.")
    parser.add_argument("file_path", type=str, help="Path to C/C++ source file to scan.")
    parser.add_argument("--provenance", type=str, default=None, help="Path to JSON file containing code provenance metadata.")
    parser.add_argument("--json", action="store_true", help="Output exact JSON results instead of human-readable report.")
    args = parser.parse_args()

    target_path = Path(args.file_path)
    if not target_path.exists():
        if args.json:
            print(json.dumps({"error": f"File not found: {target_path}"}))
        else:
            print(f"ERROR: File not found: {target_path}", file=sys.stderr)
        sys.exit(3)

    prov = None
    if args.provenance:
        prov_path = Path(args.provenance)
        if prov_path.exists():
            try:
                prov = json.loads(prov_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Warning: Failed to parse provenance JSON: {e}", file=sys.stderr)
        else:
            print(f"Warning: Provenance file not found: {prov_path}", file=sys.stderr)

    # Initialize scanner
    scanner = SecurityScanner(load_model_b=True)
    report = scanner.scan_file(target_path, provenance=prov)

    if args.json:
        print(json.dumps(report, indent=2))
        verdict = report.get("verdict", "SAFE")
        if verdict == "VULNERABLE":
            sys.exit(2)
        elif verdict == "UNSURE":
            sys.exit(1)
        else:
            sys.exit(0)

    # Human-readable format
    print("=" * 80)
    print(f"SECURITY AUDIT REPORT: {target_path.name}")
    print(f"Path: {target_path.resolve()}")
    print("=" * 80)

    print_banner(
        report["verdict"],
        report["max_p_code"],
        report["functions_scanned"][0]["calibrated_precision_at_score"] if report["functions_scanned"] else 0.0,
        report["functions_scanned"][0]["calibrated_npv_at_score"] if report["functions_scanned"] else 1.0,
    )

    print(f"Functions Analyzed: {report['num_functions']}")
    print(f"Maximum Code Risk (P_code): {report['max_p_code']:.4f}")
    if prov:
        delta_p = report['max_p_risk'] - report['max_p_code']
        print(f"Maximum Fusion Risk (P_risk): {report['max_p_risk']:.4f} (Provenance adjustment: {delta_p:+.4f})")
    print(f"Analysis Runtime: {report['runtime_ms']:.1f} ms\n")

    for idx, fn in enumerate(report["functions_scanned"]):
        fn_title = f"Function {idx + 1} (Line {fn.get('start_line', 1)})"
        print("-" * 60)
        print(f"{fn_title:<35} Verdict: [{fn['verdict']}] (P_code: {fn['p_code']:.4f}, Triage: {fn['triage_score']:.4f})")
        print(f"Length: {fn['length_tokens']} tokens ({fn['length_risk_tier']} length risk)")

        if fn.get("likely_cwe"):
            top_cwes = ", ".join([f"{c} ({p*100:.1f}%)" for c, p in fn["likely_cwe"]])
            print(f"Top Likely CWEs: {top_cwes}")

        if fn.get("static_findings"):
            print("Static Analysis Findings:")
            for s in fn["static_findings"]:
                print(f"  - [{s['tool'].upper()}] Line {s['line']}: {s['cwe']} - {s['message']}")
        else:
            print("Static Analysis: 0 tool findings (Cppcheck, Flawfinder clean)")

        if fn.get("suspicious_lines"):
            print_suspicious_lines(fn.get("code", ""), fn["suspicious_lines"], fn.get("start_line", 1))

    # Advisory Notice
    print("\n" + "=" * 80)
    print("ADVISORY NOTICE:")
    print("  Security triage assistance only. Not a formal proof of safety.")
    print("  All code requires human review before production deployment.")
    print("=" * 80 + "\n")

    # Exit code
    if report["verdict"] == "VULNERABLE":
        sys.exit(2)
    elif report["verdict"] == "UNSURE":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
